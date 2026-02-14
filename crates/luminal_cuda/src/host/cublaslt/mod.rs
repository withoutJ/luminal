use std::sync::{Arc, OnceLock};

use luminal::{
    egglog_utils::{extract_dtype, extract_expr},
    op::{
        DType, EgglogOp, LLIROp,
        OpParam::{self, *},
    },
    prelude::{
        tracing::{Level, span, trace},
        *,
    },
};

use crate::{
    cudarc::{
        cublas::sys::cublasOperation_t,
        cublaslt::{
            CudaBlasLT, MatmulShared,
            sys::{
                cublasComputeType_t, cublasLtMatmul, cublasLtMatmulAlgoGetHeuristic,
                cublasLtMatmulDesc_t, cublasLtMatmulDescCreate, cublasLtMatmulDescDestroy,
                cublasLtMatmulDescSetAttribute, cublasLtMatmulHeuristicResult_t,
                cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t,
                cublasLtMatmulPreferenceCreate, cublasLtMatmulPreferenceDestroy,
                cublasLtMatmulPreferenceSetAttribute, cublasLtMatrixLayout_t,
                cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy, cudaDataType,
            },
        },
        driver::{CudaSlice, CudaStream, DevicePtr},
    },
    host::{HostOp, cublas::parse_cublas_op},
};

#[derive(Debug)]
#[allow(dead_code)]
pub struct CuBlasLt {
    m: Expression,
    n: Expression,
    k: Expression,
    a_layout: cublasOperation_t,
    b_layout: cublasOperation_t,
    lda: Expression,
    ldb: Expression,
    ldc: Expression,
    dtype: DType,
    cublaslt: OnceLock<Arc<CudaBlasLT>>,
}

// Useless default for IntoEgglogOp
impl Default for CuBlasLt {
    fn default() -> Self {
        Self {
            m: Expression::default(),
            n: Expression::default(),
            k: Expression::default(),
            a_layout: cublasOperation_t::CUBLAS_OP_N, // IGNORE NOT REAL
            b_layout: cublasOperation_t::CUBLAS_OP_T, // IGNORE NOT REAL
            lda: Expression::default(),
            ldb: Expression::default(),
            ldc: Expression::default(),
            dtype: DType::F32,
            cublaslt: OnceLock::new(),
        }
    }
}

impl EgglogOp for CuBlasLt {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "cublaslt".to_string(),
            //    A      B      m     n      k  , A input Layout, B input Layout, lda, ldb, ldc, dtype
            vec![
                Input, Input, Expr, Expr, Expr, Str, Str, Expr, Expr, Expr, Dty,
            ],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            include_str!["cublaslt_RmRm_rewrite.egg"].to_string(), // row row
            include_str!["cublaslt_RmCm_rewrite.egg"].to_string(), // row col
            include_str!["cublaslt_CmRm_rewrite.egg"].to_string(), // col row
            include_str!["cublaslt_CmCm_rewrite.egg"].to_string(), // col col
        ]
    }

    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        // Extract dimensions from egglog
        let m = extract_expr(egraph, children[2], expr_cache).unwrap();
        let n = extract_expr(egraph, children[3], expr_cache).unwrap();
        let k = extract_expr(egraph, children[4], expr_cache).unwrap();

        // Extract layout strings from egglog
        let a_layout_str = &egraph.enodes[children[5]].0;
        let b_layout_str = &egraph.enodes[children[6]].0;
        let a_layout = parse_cublas_op(a_layout_str);
        let b_layout = parse_cublas_op(b_layout_str);

        // Extract leading dimensions from egglog
        let lda = extract_expr(egraph, children[7], expr_cache).unwrap();
        let ldb = extract_expr(egraph, children[8], expr_cache).unwrap();
        let ldc = extract_expr(egraph, children[9], expr_cache).unwrap();

        // Extract dtype from egglog
        let dtype = extract_dtype(egraph, children[10]);

        let extracted_state = Self {
            m,
            n,
            k,
            a_layout,
            b_layout,
            lda,
            ldb,
            ldc,
            dtype,
            cublaslt: OnceLock::new(),
        };
        trace!(?extracted_state);

        let extracted = LLIROp::new::<dyn HostOp>(Box::new(extracted_state) as Box<dyn HostOp>);

        (extracted, vec![children[0], children[1]])
    }

    fn cleanup(&self) -> bool {
        false
    }
}

/// Convert DType to CUDA types for cuBLAS LT
/// Returns (matrix_dtype, compute_type, scale_dtype)
fn dtype_to_cuda_types(dtype: DType) -> (cudaDataType, cublasComputeType_t, cudaDataType) {
    match dtype {
        // F32: matrix=f32, compute=f32, scale=f32
        DType::F32 => (
            cudaDataType::CUDA_R_32F,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType::CUDA_R_32F,
        ),
        // F16: matrix=f16, compute=f32 (FP32 accumulation for accuracy), scale=f32
        DType::F16 => (
            cudaDataType::CUDA_R_16F,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType::CUDA_R_32F,
        ),
        // BF16: matrix=bf16, compute=f32 with tensor cores, scale=f32
        DType::Bf16 => (
            cudaDataType::CUDA_R_16BF,
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
            cudaDataType::CUDA_R_32F,
        ),
        DType::Int => panic!("cuBLAS LT does not support integer matmul"),
        DType::Bool => panic!("cuBLAS LT does not support bool matmul"),
        DType::NvFp4 | DType::Mxfp4 => todo!("cuBLAS LT FP4 matmul not yet implemented"),
    }
}

impl HostOp for CuBlasLt {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        // GEMM parameters
        let m = self.m.exec(dyn_map).unwrap() as u64;
        let n = self.n.exec(dyn_map).unwrap() as u64;
        let k = self.k.exec(dyn_map).unwrap() as u64;
        let a_layout = self.a_layout;
        let b_layout = self.b_layout;
        let lda = self.lda.exec(dyn_map).unwrap() as i64;
        let ldb = self.ldb.exec(dyn_map).unwrap() as i64;
        let ldc = self.ldc.exec(dyn_map).unwrap() as i64;

        // Get CUDA types based on dtype
        let (cuda_dtype, compute_type, scale_dtype) = dtype_to_cuda_types(self.dtype);
        let element_size = match self.dtype {
            DType::F32 => 4u64,
            DType::F16 | DType::Bf16 => 2u64,
            DType::Int | DType::Bool => panic!("cuBLAS LT does not support integer/bool matmul"),
            DType::NvFp4 | DType::Mxfp4 => todo!("cuBLAS LT FP4 matmul not yet implemented"),
        };

        // Alpha/beta scale values (all dtypes use F32 scale type)
        let alpha_f32: f32 = 1.0;
        let beta_f32: f32 = 0.0;

        // Get buffers: output is self_node, inputs are from graph edges
        let c_buf = buffers[&self_node];
        let a_buf = buffers[&inputs[0]];
        let b_buf = buffers[&inputs[1]];

        // Get device pointers
        let (a_ptr, _a_guard) = a_buf.device_ptr(stream);
        let (b_ptr, _b_guard) = b_buf.device_ptr(stream);
        let (c_ptr, _c_guard) = c_buf.device_ptr(stream);

        // Debug tracing
        trace!(
            "buffer_validation {}=={},{}=={},{}=={}",
            a_buf.len(),
            m * k * element_size,
            b_buf.len(),
            k * n * element_size,
            c_buf.len(),
            m * n * element_size
        );
        let _span = span!(
            Level::TRACE,
            "cuBLASLT",
            m, n, k, lda, ldb, ldc, ?a_layout, ?b_layout, ?self.dtype,
        )
        .entered();

        let cublaslt = self
            .cublaslt
            .get_or_init(|| Arc::new(CudaBlasLT::new(stream.clone()).unwrap()));

        let mut matmul_desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        let mut a_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut b_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut c_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();
        let mut heuristic: cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
        let mut algo_count: i32 = 0;

        // Allocate workspace (32 MiB)
        const WORKSPACE_SIZE: usize = 32 * 1024 * 1024;
        let workspace = unsafe { stream.alloc::<u8>(WORKSPACE_SIZE)? };
        let (workspace_ptr, _workspace_guard) = workspace.device_ptr(stream);

        unsafe {
            // Create matmul descriptor (compute_type, scale_type for alpha/beta)
            cublasLtMatmulDescCreate(&mut matmul_desc, compute_type, scale_dtype).result()?;

            // Set transpose attributes
            cublasLtMatmulDescSetAttribute(
                matmul_desc,
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &a_layout as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<cublasOperation_t>(),
            )
            .result()?;
            cublasLtMatmulDescSetAttribute(
                matmul_desc,
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &b_layout as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<cublasOperation_t>(),
            )
            .result()?;

            // Create matrix layout descriptors
            let (a_rows, a_cols) = if a_layout == cublasOperation_t::CUBLAS_OP_N {
                (m, k)
            } else {
                (k, m)
            };
            let (b_rows, b_cols) = if b_layout == cublasOperation_t::CUBLAS_OP_N {
                (k, n)
            } else {
                (n, k)
            };

            cublasLtMatrixLayoutCreate(&mut a_desc, cuda_dtype, a_rows, a_cols, lda).result()?;
            cublasLtMatrixLayoutCreate(&mut b_desc, cuda_dtype, b_rows, b_cols, ldb).result()?;
            cublasLtMatrixLayoutCreate(&mut c_desc, cuda_dtype, m, n, ldc).result()?;

            // Create preference and set workspace size
            cublasLtMatmulPreferenceCreate(&mut preference).result()?;
            cublasLtMatmulPreferenceSetAttribute(
                preference,
                cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &WORKSPACE_SIZE as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
            )
            .result()?;

            // Get heuristic (best algorithm)
            cublasLtMatmulAlgoGetHeuristic(
                *cublaslt.handle(),
                matmul_desc,
                a_desc,
                b_desc,
                c_desc,
                c_desc, // D layout same as C
                preference,
                1, // Request 1 result
                &mut heuristic,
                &mut algo_count,
            )
            .result()?;

            if algo_count == 0 {
                // Cleanup before returning error
                cublasLtMatmulPreferenceDestroy(preference);
                cublasLtMatrixLayoutDestroy(c_desc);
                cublasLtMatrixLayoutDestroy(b_desc);
                cublasLtMatrixLayoutDestroy(a_desc);
                cublasLtMatmulDescDestroy(matmul_desc);
                return Err(anyhow::anyhow!("No suitable cuBLASLT algorithm found"));
            }

            // All dtypes use F32 scale type for alpha/beta
            let alpha_ptr = &alpha_f32 as *const _ as *const std::ffi::c_void;
            let beta_ptr = &beta_f32 as *const _ as *const std::ffi::c_void;
            cublasLtMatmul(
                *cublaslt.handle(),
                matmul_desc,
                alpha_ptr,
                a_ptr as *const std::ffi::c_void,
                a_desc,
                b_ptr as *const std::ffi::c_void,
                b_desc,
                beta_ptr,
                c_ptr as *const std::ffi::c_void,
                c_desc,
                c_ptr as *mut std::ffi::c_void,
                c_desc, // D layout same as C
                &heuristic.algo,
                workspace_ptr as *mut std::ffi::c_void,
                WORKSPACE_SIZE,
                stream.cu_stream() as *mut _,
            )
            .result()?;

            // Cleanup
            cublasLtMatmulPreferenceDestroy(preference);
            cublasLtMatrixLayoutDestroy(c_desc);
            cublasLtMatrixLayoutDestroy(b_desc);
            cublasLtMatrixLayoutDestroy(a_desc);
            cublasLtMatmulDescDestroy(matmul_desc);
        }

        stream.synchronize()?;
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }

    fn output_bytes(&self) -> Expression {
        let elem_size: Expression = match self.dtype {
            DType::F32 | DType::Int => 4,
            DType::F16 | DType::Bf16 => 2,
            DType::Bool => 1,
            DType::NvFp4 | DType::Mxfp4 => todo!("FP4 element size not yet implemented"),
        }
        .into();
        self.output_size() * elem_size
    }
}

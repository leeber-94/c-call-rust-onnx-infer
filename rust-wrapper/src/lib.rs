#![allow(unused_imports)]
use ndarray::{array, s, Array};
use ndarray::{Array1, Array2, Array3, ArrayD, Axis, CowArray, IxDyn};
use ort::inputs;
use ort::{ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Tensor, Value};
use rand::Error;
use serde_json::{json, Value as JsonValue};
use std::ffi::{c_char, CStr, CString};

pub fn try_load_onnx_model(model_path: &str) -> Result<ort::Session, ort::Error> {
    let model: Session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_model_from_file(model_path)?;
    println!("rust inited model.");
    Ok(model)
}

pub fn try_infer_sentence(session: &Session, content: &str) -> Result<String, ort::Error> {
    println!("rust model input: {}", content);
    let input: Array2<i32> = array![[79, 30, 73, 65, 69, 51, 57, 67]];
    let outputs = session.run(ort::inputs!["input" => input.view()]?)?;
    let output_0: Tensor<i32> = outputs["output"].extract_tensor()?;
    let output_0 = output_0.view();
    let output_0 = output_0.iter().clone().collect::<Vec<_>>();
    let output_0 = output_0.get(0).unwrap();
    let result = format!("{:?}", output_0);
    println!("rust model result: {}", result);
    Ok(result)
}

pub fn try_free_model(session: Session) {
    drop(session);
    println!("rust freed model.");
}

#[no_mangle]
pub unsafe extern "C" fn rust_try_load_onnx_model(ptr: *const c_char) -> *mut Session {
    let c_str = CStr::from_ptr(ptr);
    let rust_str = c_str.to_str().expect("Bad encoding in c_str").to_owned();
    let json_in: JsonValue = serde_json::from_str(&rust_str).unwrap();
    let model_path = json_in["model_path"].as_str().unwrap();
    let session = try_load_onnx_model(model_path).unwrap();
    Box::into_raw(Box::new(session)) // Move ownership to C
}

#[no_mangle]
pub unsafe extern "C" fn rust_try_infer_sentence(
    p_session: *mut Session,
    ptr: *const c_char,
) -> *const c_char {
    let c_str = CStr::from_ptr(ptr);
    let rust_str = c_str.to_str().expect("Bad encoding in c_str").to_owned();
    let json_in: JsonValue = serde_json::from_str(&rust_str).unwrap();
    let sentence = json_in["sentence"].as_str().unwrap();

    let session = unsafe {
        assert!(!p_session.is_null());
        &*(p_session) // not Move ownership
    };
    let result = try_infer_sentence(session, sentence).expect("模型预测出错");
    let data = json!({"result":result}).to_string();
    let c_string: CString = CString::new(data).expect("CString::new failed");
    c_string.into_raw() // Move ownership to C
}

#[no_mangle]
pub unsafe extern "C" fn rust_try_free_model(p_session: *mut Session) {
    let session = unsafe {
        assert!(!p_session.is_null());
        *Box::from_raw(p_session) // Move ownership to rust
    };
    try_free_model(session);
}

#[no_mangle]
pub unsafe extern "C" fn rust_free_string(ptr: *const c_char) {
    let _ = CString::from_raw(ptr as *mut _); // Move ownership to rust
}

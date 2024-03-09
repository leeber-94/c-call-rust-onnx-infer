use rust_wrapper::*;

fn main() {
    let session = try_load_onnx_model("../simple_model.onnx").expect("load model error");
    let sentence = "假设这个是测试文本";
    let result: String = try_infer_sentence(&session,sentence).expect("模型预测出错");
    println!("{}",result);
    try_free_model(session);
}

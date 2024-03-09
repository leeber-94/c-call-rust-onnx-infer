#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"

typedef struct OnnxModel_S OnnxModel_t;
extern OnnxModel_t *rust_try_load_onnx_model(char *);
extern const char *rust_try_infer_sentence(OnnxModel_t *, char *);
extern void rust_try_free_model(OnnxModel_t *);
extern void rust_free_string(const char *);

int main()
{
    // init-model
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "model_path", "../simple_model.onnx");
    OnnxModel_t *onnxModel = rust_try_load_onnx_model(cJSON_PrintUnformatted(root));
    cJSON_Delete(root);


    // infer-one sentence
    cJSON *infer_data = cJSON_CreateObject();
    cJSON_AddStringToObject(infer_data, "sentence", "假设这个是测试文本");
    printf("send to rust:%s\n", cJSON_PrintUnformatted(infer_data));
    const char *rust_result = rust_try_infer_sentence(onnxModel, cJSON_PrintUnformatted(infer_data));
    printf("get from rust:%s\n", rust_result);
    rust_free_string(rust_result);
    cJSON_Delete(infer_data);

    // free-model
    rust_try_free_model(onnxModel);

    return 0;
}
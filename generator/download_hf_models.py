from huggingface_hub import snapshot_download


def get_hf_model(repo_name, cache="D:\study\huggingface\caches", use_auth_token=None, revision=None):
    if revision is None:
        return snapshot_download(repo_name, cache_dir=cache, use_auth_token=use_auth_token,
                                 )
    else:
        return snapshot_download(repo_name, cache_dir=cache, use_auth_token=use_auth_token, revision=revision,
                                 )

#

if __name__ == '__main__':
    print(get_hf_model("Salesforce/codegen-2B-multi"))
    print(get_hf_model("Salesforce/codegen-6B-multi"))
    print(get_hf_model("bigcode/starcoder2-3b", use_auth_token="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"))
    print(get_hf_model("bigcode/starcoder2-7b", use_auth_token="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"))

    print(get_hf_model("facebook/incoder-1B", use_auth_token="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"))
    print(get_hf_model("facebook/incoder-6B", use_auth_token="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"))






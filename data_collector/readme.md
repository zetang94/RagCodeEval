```mermaid
graph TD;
	HistoryId[历史提交ID]--git.reset-->Repo[历史代码库];
	Repo --SG.init and index--> StackGraph;
	AddMethod[新开发代码] --treesitter--> Calls[寻找到有方法调用的代码];
	AddMethod --git.reset and SG.index --> USG[更新调用图]
	Calls --寻找方法定义 --> FuncDef[方法定义, 文件路径 line col 名称]
	USG --> FuncDef
	StackGraph --SG.query--> Callgraph[被调用函数id, 调用者函数id列表]
	FuncDef --绑定历史代码库--> HistoryFuncDef[如果文件路径没有被修改,去历史库中按照line col查找. 否则,去历史库中的文件路径对应代码中,按照callname查询]
	HistoryFuncDef --> HistoryFuncID[被调用函数的id]
```

## Test dataset builder
The selected repos must satisfy the following requirements:
    1. ignore repos with less than 50 methods.
    2. file nums <= 1000. (For BM25 search)
    3. commit代码文件个数 > 10

The added methods should not be same before the same repo, (refactor)

The called func must satisfy the following requirements:
    1. does not occur before in current file.
    2. has either cross-file definition or history callee in the repository. (This makes search action meaningful.)
    3. There are no two same ground truths in the same repository.
    4. only contains one call in the target line (To avoid complex retrieval conditions.)

## Test Sample Format
```json
{   
    "prompt": "the left code context",
    "groundtruth": "the target completed code",
     
    "call_meta": {
        "signature": "func_name:param_list",
        
        "called_def": {
            "stmt": "the definition stmt", 
            "file_path": " the location of the def",
            "type": "method/class",
                # To evaluate the acc of different retrieval methods,
                # for class def, its range is [-1, -1], which means the whole code file.
                # for method def, its range is [start_pos, end_pos]
            "range": [[line_no, col_no], [line_no, col_no]]
         },
         
        "history_callees": [
            {"file_path": "the location of history callee",
             "callee_stmt": "the stmt of callee",
             "range": [[line_no, col_no], [line_no, col_no]]}
        ],
    },
    
    "file_meta": {
        "file_path": "",
        "commit_id": "",
        "lifecycle": "Initiation/Intermediate/Closure",
        "delete_import_lines": [(line_no, import_stmt)],
        "right_context": "",
    }
}
```


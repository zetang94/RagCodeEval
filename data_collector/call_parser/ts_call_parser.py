from tree_sitter import Language, Parser

from .base_call_parser import CallParser


class TSCallParser(CallParser):
    language = "TypeScript"
    extension = '.ts'
    language_library = Language('./call_parser/my-languages.so', 'typescript')
    parser = Parser()
    parser.set_language(language_library)

    query = language_library.query("""
                (
                  (call_expression
                    function: (identifier) @func_call)
                )
                (call_expression
                  function: (member_expression
                    property: (property_identifier) @func_call)
                ) 
                (new_expression
                  constructor: (_) @construct_call)

                (import_statement) @import
            """)


if __name__ == '__main__':
    code = """
function add(a: number, b: number): number {
    return a + b;
}

console.log("Result of add(2, 3):", add(2, 3));
function normalFunction() {
    console.log("Normal function call");
}

class MyClass {
    static staticMethod() {
        console.log("Static method call");
    }

    instanceMethod() {
        console.log("Instance method call");
    }
}

const myObject = new MyClass();
normalFunction();
myObject.instanceMethod();
MyClass.staticMethod();
new MyClass();
(() => {})();
    """

    call_parser = TSCallParser()
    result = call_parser.extract(code)

    print(result)
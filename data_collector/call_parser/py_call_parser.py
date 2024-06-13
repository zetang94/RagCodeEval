from tree_sitter import Language, Parser

from .base_call_parser import CallParser


class PyCallParser(CallParser):
    language = "Python"
    extension = '.py'
    language_library = Language('./call_parser/my-languages.so', 'python')
    parser = Parser()
    parser.set_language(language_library)

    query = language_library.query("""
        (call
          function: [
              (identifier) @static_or_construct_call
              (attribute
                attribute: (identifier) @func_call)
          ])
          
        (import_statement) @import 
        (import_from_statement) @import 
    """)


if __name__ == "__main__":
    code = """
import c
from d import e

class A:
    e=None
    
    def b():
        pass
    
    def c():
        pass
    
    
a = d.A()
A.b()
a.e
A.e
a.c(f,g)
    """

    parser = PyCallParser()
    r = parser.extract(code)

    for i in r:
        print(i)




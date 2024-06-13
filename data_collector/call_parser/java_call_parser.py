from tree_sitter import Language, Parser
from .base_call_parser import CallParser


class JavaCallParser(CallParser):
    language = "Java"
    extension = '.java'
    language_library = Language('./call_parser/my-languages.so', 'java')
    parser = Parser()
    parser.set_language(language_library)

    query = language_library.query("""
        (object_creation_expression
          type: (type_identifier) @construct_call)
          
        (method_invocation
          name: (identifier) @func_call
          arguments: (argument_list))
          
       (import_declaration) @import
    """)





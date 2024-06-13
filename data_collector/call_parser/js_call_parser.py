from tree_sitter import Language, Parser

from .base_call_parser import CallParser


class JSCallParser(CallParser):
    language = "JavaScript"
    extension = '.js'
    language_library = Language('./call_parser/my-languages.so', 'javascript')
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


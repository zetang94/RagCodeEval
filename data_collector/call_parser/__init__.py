from .java_call_parser import JavaCallParser
from .py_call_parser import PyCallParser
from .js_call_parser import JSCallParser
from .ts_call_parser import TSCallParser

langs = {
    JavaCallParser.language: JavaCallParser(),
    PyCallParser.language: PyCallParser(),
    JSCallParser.language: JSCallParser(),
    TSCallParser.language: TSCallParser()
}


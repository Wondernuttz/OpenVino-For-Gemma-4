"""CPU-only contract checks; no OpenVINO install or GPU/model loading required."""
import ast
import importlib.util
from pathlib import Path
import re
import types
import unittest
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location('launch', ROOT / 'launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)

def helpers():
    tree = ast.parse((ROOT / 'ovserver_moe.py').read_text())
    wanted = {'build_prompt', 'strip_thinking', 'clean_visible_answer', 'extract_reasoned_answer',
              'cap_context', 'ReasoningBoundaryError'}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {'_GEMMA_THOUGHT', '_XML_THOUGHT', '_THOUGHT_LINE', '_REASONING_MARKER'} for t in n.targets)]
    ns = {'re': re, 'FMT': 'gemma', 'THINK': False, 'MAX_CTX_TOKENS': 220,
          '_MIN_OUTPUT_RESERVE': 32, '_CTX_MARGIN_TOKENS': 8}
    ns['TOK'] = types.SimpleNamespace(encode=lambda text, **kw:
        types.SimpleNamespace(input_ids=types.SimpleNamespace(shape=(1, len(text)))))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'helpers', 'exec'), ns)
    return ns

class PortableTests(unittest.TestCase):
    def test_default_12b(self):
        env = launch.configure({'OV_MODEL': '/model'})
        self.assertEqual(env['OV_PROFILE'], 'gemma12-text')
        self.assertEqual(env['OV_DQGS'], '128')
        self.assertEqual(env['OV_PREFIX_CACHE'], '0')
        self.assertEqual(env['OV_CONTEXT_TOKENS'], '4096')

    def test_26b_profile(self):
        env = launch.configure({'OV_MODEL': '/model', 'OV_PROFILE': 'gemma26-b70'})
        for k,v in {'OV_DQGS':'128','OV_PREFIX_CACHE':'8','OV_MAX_BATCHED_TOKENS':'16384',
                    'OV_KV_CACHE_PRECISION':'u4','MOE_GROUPED_BINARY_LOOKUP':'1'}.items():
            self.assertEqual(env[k],v)

    def test_invalid_profiles(self):
        for env in ({}, {'OV_MODEL':'/m','OV_PROFILE':'oops'},
                    {'OV_MODEL':'/m','OV_PREFIX_CACHE':'8'},
                    {'OV_MODEL':'/m','OV_CONTEXT_TOKENS':'512'}):
            with self.assertRaises(ValueError): launch.configure(env)

    def test_client_override(self):
        self.assertEqual(launch.configure({'OV_MODEL':'/m','OV_DQGS':'0'})['OV_DQGS'],'0')

    def test_reasoning_closed(self):
        ns=helpers()
        self.assertEqual(ns['extract_reasoned_answer']('<|channel>thought\nsecret<channel|>Hello<turn|>'),'Hello')

    def test_reasoning_unclosed(self):
        self.assertIsNone(helpers()['extract_reasoned_answer']('<|channel>thought\nsecret'))

    def test_reasoning_residue_rejected(self):
        ns=helpers()
        with self.assertRaises(ns['ReasoningBoundaryError']):
            ns['clean_visible_answer']('hello <think>secret')

    def test_context_slides(self):
        ns=helpers()
        messages=[{'role':'system','content':'keep me'}, {'role':'user','content':'old'*100},
                  {'role':'assistant','content':'older'}, {'role':'user','content':'new'}]
        result=ns['cap_context'](messages)
        self.assertEqual(result[0], messages[0])
        self.assertEqual(result[-1], messages[-1])
        self.assertLessEqual(len(ns['build_prompt'](result)),180)

    def test_system_overflow_rejected(self):
        with self.assertRaises(ValueError):
            helpers()['cap_context']([{'role':'system','content':'x'*1000}])

    def test_no_machine_defaults(self):
        for name in ['ovserver_moe.py','start-ov-8002-buddy.sh','start-ov-8092-bots.sh']:
            text=(ROOT/name).read_text()
            self.assertNotIn('/home/wondernutts',text)
            self.assertNotIn('pkill',text)

class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse((ROOT/'ovserver_moe.py').read_text())
        handler=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='H')
        ns={'BaseHTTPRequestHandler':BaseHTTPRequestHandler,'json':json,'hmac':hmac,
            'API_KEY':'test-key','MODEL_NAME':'test-model','DEVICE':'TEST','time':time,
            '_has_vision':False,'_has_audio':False,'_vision_reason':'disabled',
            'AttachmentError':ValueError,'generate':lambda messages, req:'Hello traveler.'}
        exec(compile(ast.Module(body=[handler],type_ignores=[]),'handler','exec'),ns)
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),ns['H'])
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url='http://127.0.0.1:'+str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def request(self,path,body=None,auth=True):
        headers={'Content-Type':'application/json'}
        if auth: headers['Authorization']='Bearer test-key'
        req=Request(self.url+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
        try:
            with urlopen(req,timeout=3) as response: return response.status,response.read()
        except HTTPError as e:
            with e: return e.code,e.read()

    def test_health(self): self.assertEqual(self.request('/health',auth=False)[0],200)
    def test_auth(self): self.assertEqual(self.request('/v1/models',auth=False)[0],401)
    def test_chat(self):
        code,body=self.request('/v1/chat/completions',{'model':'test-model','messages':[{'role':'user','content':'hi'}]})
        self.assertEqual(code,200)
        self.assertEqual(json.loads(body)['choices'][0]['message']['content'],'Hello traveler.')
        self.assertNotIn('usage',json.loads(body))
    def test_wrong_model(self):
        self.assertEqual(self.request('/v1/chat/completions',{'model':'wrong'})[0],404)
    def test_non_object(self): self.assertEqual(self.request('/v1/chat/completions',[])[0],400)
    def test_image_rejected(self):
        self.assertEqual(self.request('/v1/chat/completions',{'messages':[{'role':'user','content':[{'type':'image_url','image_url':'http://localhost'}]}]})[0],400)
    def test_av_rejected(self): self.assertEqual(self.request('/av',{'kind':'image'})[0],409)
    def test_tools_rejected(self): self.assertEqual(self.request('/v1/chat/completions',{'tools':[{}]})[0],400)
    def test_buffered_sse(self):
        code,body=self.request('/v1/chat/completions',{'messages':[{'role':'user','content':'hi'}],'stream':True})
        self.assertEqual(code,200); self.assertIn(b'data: [DONE]',body)

if __name__ == '__main__': unittest.main()

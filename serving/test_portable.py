"""CPU-only contract checks; no OpenVINO install or GPU/model loading required."""
import ast
import importlib.util
from pathlib import Path
import re
import types
import unittest
import hmac
import json
import math
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
              'cap_context', 'ReasoningBoundaryError', 'performance_attempt',
              'measured_generate', 'performance_report', 'generate'}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {'_GEMMA_THOUGHT', '_XML_THOUGHT', '_THOUGHT_LINE', '_REASONING_MARKER'} for t in n.targets)]
    ns = {'re': re, 'math': math, 'time': time, '_PFX_GB': 0,
          'FMT': 'gemma', 'THINK': False, 'MAX_CTX_TOKENS': 220,
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

    def test_wideq_opt_in(self):
        env=launch.configure({'OV_MODEL':'/m','OV_PROFILE':'gemma26-b70','GEMMA_MIXED_512_TILE':'wideq'})
        self.assertEqual(env['GEMMA_MIXED_512_TILE'],'wideq')
        self.assertNotIn('GEMMA_MIXED_512_TILE',launch.configure({'OV_MODEL':'/m','OV_PROFILE':'gemma26-b70'}))

    def test_wideq_rejects_12b_and_experiments(self):
        for profile,tile in [('gemma12-text','wideq'),('gemma26-b70','64'),('gemma26-b70','compact')]:
            with self.assertRaises(ValueError):
                launch.configure({'OV_MODEL':'/m','OV_PROFILE':profile,'GEMMA_MIXED_512_TILE':tile})

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
            'AttachmentError':ValueError,'generate':lambda messages, req:(
                'Hello traveler.', {'schema_version':1,'attempts':[{'generated_tokens':3}]})}
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
        self.assertEqual(json.loads(body)['performance']['attempts'][0]['generated_tokens'],3)
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
        chunks=[json.loads(line[6:]) for line in body.decode().splitlines()
                if line.startswith('data: {')]
        self.assertNotIn('performance',chunks[0])
        self.assertEqual(chunks[-1]['performance']['schema_version'],1)


class PerformanceTests(unittest.TestCase):
    def result(self, inputs=2048, outputs=64, ttft=400, throughput=58):
        return types.SimpleNamespace(perf_metrics=types.SimpleNamespace(
            get_num_input_tokens=lambda: inputs,
            get_num_generated_tokens=lambda: outputs,
            get_ttft=lambda: types.SimpleNamespace(mean=ttft),
            get_throughput=lambda: types.SimpleNamespace(mean=throughput)))

    def test_native_metrics(self):
        row=helpers()['performance_attempt'](self.result(),2,'answer')
        self.assertEqual(row['pp_tokens_per_ttft_second'],5120)
        self.assertEqual(row['decode_tokens_per_second'],58)
        self.assertEqual(row['generated_tokens'],64)

    def test_no_metrics(self):
        row=helpers()['performance_attempt']('a plain string',2,'answer')
        for key in ['input_tokens','generated_tokens','ttft_ms',
                    'pp_tokens_per_ttft_second','decode_tokens_per_second']:
            self.assertIsNone(row[key])

    def test_invalid_native_values(self):
        for invalid in [float('nan'),float('inf'),-1]:
            row=helpers()['performance_attempt'](self.result(invalid,invalid,invalid,invalid),2,'answer')
            self.assertIsNone(row['input_tokens'])
            self.assertIsNone(row['generated_tokens'])
            self.assertIsNone(row['ttft_ms'])
            json.dumps(row,allow_nan=False)

    def test_zero_ttft(self):
        row=helpers()['performance_attempt'](self.result(ttft=0),2,'answer')
        self.assertIsNone(row['pp_tokens_per_ttft_second'])

    def test_single_output(self):
        row=helpers()['performance_attempt'](self.result(outputs=1),2,'answer')
        self.assertIsNone(row['decode_tokens_per_second'])

    def test_cached_pp_suppressed(self):
        ns=helpers(); ns['_PFX_GB']=8
        row=ns['performance_attempt'](self.result(),2,'answer')
        self.assertIsNone(row['pp_tokens_per_ttft_second'])
        self.assertEqual(row['input_tokens'],2048)

    def test_getter_failure(self):
        result=self.result()
        def broken(): raise RuntimeError('backend has no metric')
        result.perf_metrics.get_ttft=broken
        row=helpers()['performance_attempt'](result,2,'answer')
        self.assertIsNone(row['ttft_ms'])
        self.assertEqual(row['generated_tokens'],64)

    def generation_ns(self, thinking=False, retry=False):
        ns=helpers(); calls=[]
        class Collector:
            toks=[1,2,3]
        ns.update(THINK=thinking, make_cfg=lambda *a,**k:types.SimpleNamespace(max_new_tokens=64),
                  cap_context=lambda messages,**k:messages,
                  prepare_vision_messages=lambda m:(m,[]),
                  build_prompt=lambda *a,**k:'unchanged prompt',
                  GEN_LOCK=threading.Lock(), _Collector=Collector,
                  TOK=types.SimpleNamespace(decode=lambda *a,**k:'raw thought'),
                  extract_reasoned_answer=lambda text:None if retry else 'Visible answer.',
                  _result_text=lambda result:'Visible answer.',
                  clean_visible_answer=lambda text:text)
        def pipe(prompt,cfg,images,**kwargs):
            calls.append((prompt,cfg,images,kwargs))
            return self.result(outputs=64 if not calls[:-1] else 10)
        ns['_pipe_generate']=pipe
        return ns,calls

    def test_generation_unchanged(self):
        ns,calls=self.generation_ns()
        text,perf=ns['generate']([], {})
        self.assertEqual(text,'Visible answer.')
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],'unchanged prompt')
        self.assertEqual(calls[0][3],{}) # no new streamer/extra inference
        self.assertEqual(perf['attempts'][0]['kind'],'answer')
        self.assertEqual(perf['retry_count'],0)

    def test_thinking_includes_hidden_counts(self):
        ns,calls=self.generation_ns(thinking=True)
        text,perf=ns['generate']([], {})
        self.assertEqual(text,'Visible answer.')
        self.assertEqual(len(calls),1)
        self.assertEqual(perf['attempts'][0]['generated_tokens'],64)
        self.assertTrue(perf['generated_tokens_include_hidden_reasoning'])
        self.assertIn('streamer',calls[0][3])

    def test_retry_separate_not_overwritten(self):
        ns,calls=self.generation_ns(thinking=True,retry=True)
        text,perf=ns['generate']([], {})
        self.assertEqual(text,'Visible answer.')
        self.assertEqual(len(calls),2)
        self.assertEqual(perf['retry_count'],1)
        self.assertEqual([a['generated_tokens'] for a in perf['attempts']],[64,10])
        self.assertEqual(perf['attempts'][1]['kind'],'no_think_retry')

if __name__ == '__main__': unittest.main()

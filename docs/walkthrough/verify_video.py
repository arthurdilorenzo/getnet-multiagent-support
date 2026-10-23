import hashlib,json,pathlib,re,subprocess,os,shutil
from PIL import Image,ImageDraw
O=pathlib.Path(__file__).resolve().parent;FF=os.environ.get('FFMPEG') or shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg';FP=os.environ.get('FFPROBE') or shutil.which('ffprobe') or '/opt/homebrew/bin/ffprobe';V=O/'Getnet_Walkthrough_Tecnico.mp4'
probe=json.loads(subprocess.check_output([FP,'-v','error','-show_streams','-show_format','-of','json',str(V)],text=True))
assert 180<=float(probe['format']['duration'])<=300
assert V.stat().st_size<25_000_000
assert {x['codec_type'] for x in probe['streams']}=={'audio','video'}
subprocess.run([FF,'-v','error','-i',str(V),'-f','null','-'],check=True,capture_output=True)
volume=subprocess.run([FF,'-i',str(V),'-vn','-af','volumedetect','-f','null','-'],capture_output=True,text=True,check=True).stderr
levels={k:float(v) for k,v in re.findall(r'(mean_volume|max_volume): ([-\d.]+) dB',volume)}
assert levels['mean_volume']>-45
(O/'frames').mkdir(exist_ok=True)
meta=json.loads((O/'sources.json').read_text());sheet=Image.new('RGB',(960,720),'#10151d');draw=ImageDraw.Draw(sheet)
for j,s in enumerate(meta['scenes']):
 frame=O/'frames'/f'verified-{j+1:02}.png'
 subprocess.run([FF,'-y','-v','error','-ss',str(s['start']+min(3,s['duration']/2)),'-i',str(V),'-frames:v','1',str(frame)],check=True)
 im=Image.open(frame).convert('RGB');im.thumbnail((320,180));sheet.paste(im,((j%3)*320,(j//3)*180))
sheet.save(O/'contact-sheet.jpg')
receipt={'video':V.name,'bytes':V.stat().st_size,'sha256':hashlib.sha256(V.read_bytes()).hexdigest(),'duration_seconds':float(probe['format']['duration']),'streams':[{'type':s['codec_type'],'codec':s['codec_name'],'duration':s.get('duration'),'sample_rate':s.get('sample_rate'),'channels':s.get('channels'),'width':s.get('width'),'height':s.get('height'),'r_frame_rate':s.get('r_frame_rate')} for s in probe['streams']],'audio_volume_db':levels,'full_decode':'passed','sampled_frames':11,'network_calls':0,'limits':'Não houve escuta humana; verificação de áudio por stream, duração, decodificação e nível. Cenas de código/diagramas, sem gravação UI.'}
(O/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False,indent=2))

"""Execute safe defaults and generate self-contained previews."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from html import escape
import mistune
root=Path(__file__).resolve().parents[1]
for path in sorted((root/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code': cell.outputs=[]; cell.execution_count=None
        cell.metadata={}
    nb.metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
    for cell in nb.cells: cell.metadata={}
    nb.metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}
    nbformat.write(nb,path)
    sections=[]
    for cell in nb.cells:
        if cell.cell_type=='markdown':
            sections.append(mistune.html(cell.source))
        else:
            sections.append('<pre class="code"><code>'+escape(cell.source)+'</code></pre>')
            for out in cell.outputs:
                text=out.get('text',out.get('data',{}).get('text/plain',''))
                sections.append('<pre class="output">'+escape(text)+'</pre>')
    html='''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HAR notebook preview</title><style>
body{max-width:880px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#172b3a;background:#fcfcfa}
h1,h2{line-height:1.2;color:#123f55}h2{margin-top:36px}
pre{padding:18px;border-radius:8px;overflow:auto;font:14px/1.5 monospace}
.code{background:#edf1f4}.output{background:#e9f5ef;border-left:4px solid #34805b}
a{color:#126788}code{font-size:.9em}</style><main>'''+''.join(sections)+'</main></html>'
    path.with_suffix('.html').write_text(html,encoding='utf-8')
    print(path.name+': executed')

"""Assemble Manuscript_Neurocomputing_R2: insert the section drafts, fill every number from r2_stats (strict),
number references by first appearance, and convert to .docx with the R1 manuscript as style reference."""
import re, sys, subprocess
sys.path.insert(0,'/home/claude/kt/src'); import r2_stats as RS
P='/home/claude/kt/paper'
NEW={
 'simplekt':'Z. Liu, Q. Liu, J. Chen, S. Huang, W. Luo, simpleKT: a simple but tough-to-beat baseline for knowledge tracing, in: Int. Conf. Learning Representations (ICLR 2023), 2023. arXiv:2302.06881.',
 'rkt':'S. Pandey, J. Srivastava, RKT: relation-aware self-attention for knowledge tracing, in: Proc. 29th ACM Int. Conf. Information and Knowledge Management (CIKM 2020), ACM, 2020, pp. 1205–1214. https://doi.org/10.1145/3340531.3411994',
 'pdkt':'P. Chen, Y. Lu, V.W. Zheng, Y. Pian, Prerequisite-driven deep knowledge tracing, in: Proc. 2018 IEEE Int. Conf. Data Mining (ICDM), IEEE, 2018, pp. 39–48. https://doi.org/10.1109/ICDM.2018.00019',
 'cskt':'Y. Bai, X. Li, Z. Liu, Y. Huang, T. Guo, M. Hou, F. Xia, W. Luo, csKT: addressing cold-start problem in knowledge tracing via kernel bias and cone attention, Expert Syst. Appl. 266 (2025) 125988. https://doi.org/10.1016/j.eswa.2024.125988',
 'datashop':'K.R. Koedinger, R.S.J.d. Baker, K. Cunningham, A. Skogsholm, B. Leber, J. Stamper, A data repository for the EDM community: the PSLC DataShop, in: C. Romero, S. Ventura, M. Pechenizkiy, R.S.J.d. Baker (Eds.), Handbook of Educational Data Mining, CRC Press, Boca Raton, 2010, pp. 65–78. https://doi.org/10.1201/b10274-10',
}
def r1_refs():
    out={}
    for blk in open(f'{P}/r1_refs.md').read().split('\n\n'):
        m=re.match(r'\\\[(\d+)\\\]\s+(.*)',blk.strip(),re.S)
        if m: out['r'+m.group(1)]=' '.join(m.group(2).split())
    return out
REFS={**r1_refs(),**NEW}
REFS['r39']='H.B. Truong, fuzzy-kg-modularity, version 2.0.0 [software], Zenodo, 2026. https://doi.org/10.5281/zenodo.23830895'
def number_refs(text):
    order=[]
    for m in re.finditer(r'\[\[([a-z0-9,]+)\]\]',text):
        for k in m.group(1).split(','):
            if k not in REFS: raise KeyError(f'unknown reference key {k}')
            if k not in order: order.append(k)
    num={k:i+1 for i,k in enumerate(order)}
    def fmt(m):
        ns=sorted(num[k] for k in m.group(1).split(',')); parts=[]; i=0
        while i<len(ns):
            j=i
            while j+1<len(ns) and ns[j+1]==ns[j]+1: j+=1
            parts.append(str(ns[i]) if j==i else (f'{ns[i]},{ns[j]}' if j==i+1 else f'{ns[i]}–{ns[j]}')); i=j+1
        return '['+','.join(parts)+']'
    text=re.sub(r'\[\[([a-z0-9,]+)\]\]',fmt,text)
    reflist='\n\n'.join(f'[{num[k]}] {REFS[k]}' for k in order)
    return text, reflist, order
def build(dynamic):
    S=RS.S
    S['RELATED']=open(f'{P}/r2_related_setup.md').read().split('## 5. Experimental setup')[0].strip()
    S['SETUP']='## 5. Experimental setup'+open(f'{P}/r2_related_setup.md').read().split('## 5. Experimental setup')[1].rstrip()
    S['METHOD']=open(f'{P}/r2_method.md').read().strip()
    S['SETUP']=S['SETUP'].replace('TABLE1_HERE',S['TABLE1'])
    for k,v in dynamic.items(): S[k]=RS.render(v)
    S['REFERENCES']='{REFERENCES}'
    text=RS.render(open(f'{P}/R2_template.md').read())
    text,reflist,order=number_refs(text)
    text=text.replace('{REFERENCES}',reflist.replace('[','\\[').replace(']','\\]'))
    open(f'{P}/Manuscript_Neurocomputing_R2.md','w').write(text)
    subprocess.run(['pandoc',f'{P}/Manuscript_Neurocomputing_R2.md','-o',f'{P}/Manuscript_Neurocomputing_R2.docx','--reference-doc=/home/claude/Manuscript_Neurocomputing_R1.docx',f'--resource-path={P}'],check=True)
    return text,order
if __name__=='__main__':
    from r2_dynamic import DYN
    t,o=build(DYN); print(len(t.split()),'words;',len(o),'references')

def polish(path):
    """Tables: 7.5 pt text, single spacing, first column wider; keep pandoc's structure."""
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    d=Document(path)
    sec=d.sections[0]; usable=sec.page_width-sec.left_margin-sec.right_margin
    for t in d.tables:
        n=len(t.columns)
        tblPr=t._tbl.tblPr
        lay=OxmlElement('w:tblLayout'); lay.set(qn('w:type'),'fixed'); tblPr.append(lay)
        first=int(usable*(0.22 if n>6 else 0.30)); rest=int((usable-first)/(n-1))
        grid=t._tbl.tblGrid
        for i,gc in enumerate(grid.findall(qn('w:gridCol'))): gc.set(qn('w:w'),str(int((first if i==0 else rest)/635)))
        for row in t.rows:
            for i,c in enumerate(row.cells):
                c.width=first if i==0 else rest
                for p in c.paragraphs:
                    p.paragraph_format.space_after=Pt(0); p.paragraph_format.space_before=Pt(0); p.paragraph_format.line_spacing=1.0
                    for r in p.runs: r.font.size=Pt(7.5 if n>6 else 8.5)
    d.save(path)
if __name__=='__main__':
    polish(f'{P}/Manuscript_Neurocomputing_R2.docx'); print('polished')

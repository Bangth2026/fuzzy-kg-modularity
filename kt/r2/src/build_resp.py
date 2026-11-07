"""Render the Vietnamese response table from its template with the same numbers as the manuscript."""
import subprocess, r2_stats as RS
from r2_dynamic import DYN
from build_r2 import polish
P='/home/claude/kt/paper'
for k,v in DYN.items(): RS.S[k]=RS.render(v)
open(f'{P}/Bang_doi_chieu_phan_bien_R2.md','w').write(RS.render(open(f'{P}/R2_response_template.md').read()))
subprocess.run(['pandoc',f'{P}/Bang_doi_chieu_phan_bien_R2.md','-o',f'{P}/Bang_doi_chieu_phan_bien_R2.docx','--reference-doc=/home/claude/Manuscript_Neurocomputing_R1.docx'],check=True)
polish(f'{P}/Bang_doi_chieu_phan_bien_R2.docx')

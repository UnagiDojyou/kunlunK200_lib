from pathlib import Path
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';P=R/'repos/llama.cpp/examples/kunlun'
for name,newname in [('kunlun-flash-decode-kq','kunlun-flash-decode-kq-head'),('kunlun-flash-decode-pv-head','kunlun-flash-decode-pv-head-affinity'),('kunlun-flash-profile-decode-kq','kunlun-flash-profile-decode-kq-head'),('kunlun-flash-profile-decode-pv','kunlun-flash-profile-decode-pv-head')]:
 p=L/'kernels'/(name+'.xpu');s=p.read_text().replace(name.replace('-','_'),newname.replace('-','_'))
 if 'kq' in name:
  old='c=unit%fd.chunks,hk=unit/fd.chunks%p.hk,b=unit/fd.chunks/p.hk;';new='c=unit/(p.b*p.hk),hk=unit%p.hk,b=unit/p.hk%p.b;'
 else:old='c=task%fd.chunks,hk=task/fd.chunks%p.hk,b=task/fd.chunks/p.hk;';new='c=task/(p.b*p.hk),hk=task%p.hk,b=task/p.hk%p.b;'
 assert old in s;s=s.replace(old,new);s='// Head affinity: retain the legacy head-to-SDNN assignment when heads already fill all four arrays.\n'+s
 (L/'kernels'/(newname+'.xpu')).write_text(s)
p=L/'src/flash.inc';s=p.read_text()
for old,new in [('decode_kq','decode_kq_head'),('decode_pv_head','decode_pv_head_affinity'),('profile_decode_kq','profile_decode_kq_head'),('profile_decode_pv','profile_decode_pv_head')]:
 s=s.replace(old+',',old+','+new+',',1);s=s.replace('&'+old+',','&'+old+',&'+new+',',1)
p.write_text(s)
p=L/'src/flash-decode.inc';s=p.read_text();needle='auto &online=profile?'
s=s.replace(needle,'bool affinity=head&&std::getenv("KUNLUN_FLASH_DECODING_WORK")&&!std::strcmp(std::getenv("KUNLUN_FLASH_DECODING_WORK"),"head");\n '+needle)
s=s.replace('auto &pv=profile?s.profile_decode_pv:(head?s.decode_pv_head:s.decode_pv);','auto &pv=profile?(affinity?s.profile_decode_pv_head:s.profile_decode_pv):(head?(affinity?s.decode_pv_head_affinity:s.decode_pv_head):s.decode_pv);')
s=s.replace('auto &kq_program=profile?s.profile_decode_kq:s.decode_kq;','auto &kq_program=profile?(affinity?s.profile_decode_kq_head:s.profile_decode_kq):(affinity?s.decode_kq_head:s.decode_kq);')
s=s.replace('profile?"kunlun-flash-profile-decode-kq":"kunlun-flash-decode-kq"','profile?(affinity?"kunlun-flash-profile-decode-kq-head":"kunlun-flash-profile-decode-kq"):(affinity?"kunlun-flash-decode-kq-head":"kunlun-flash-decode-kq")')
s=s.replace('profile?"kunlun-flash-profile-decode-pv":(head?"kunlun-flash-decode-pv-head":"kunlun-flash-decode-pv")','profile?(affinity?"kunlun-flash-profile-decode-pv-head":"kunlun-flash-profile-decode-pv"):(head?(affinity?"kunlun-flash-decode-pv-head-affinity":"kunlun-flash-decode-pv-head"):"kunlun-flash-decode-pv")')
s=s.replace('layout=%s calls=%zu','layout=%s work=%s calls=%zu').replace('head?"head":"chunk",s.calls','head?"head":"chunk",affinity?"head":"chunk",s.calls');p.write_text(s)
p=L/'CMakeLists.txt';s=p.read_text().replace('foreach(name kunlun-flash-profile-', 'foreach(name kunlun-flash-decode-kq-head kunlun-flash-decode-pv-head-affinity kunlun-flash-profile-decode-kq-head kunlun-flash-profile-decode-pv-head kunlun-flash-profile-',1);p.write_text(s)
p=L/'bench/flash-decode-bench.cpp';s=p.read_text();old='auto oldrun=[&]{bool compare_layout=getenv("KUNLUN_FLASH_DECODING_COMPARE_LAYOUT");'
s=s.replace(old,'auto oldrun=[&]{bool compare_work=getenv("KUNLUN_FLASH_DECODING_COMPARE_WORK");std::string work=getenv("KUNLUN_FLASH_DECODING_WORK")?getenv("KUNLUN_FLASH_DECODING_WORK"):"chunk";if(compare_work)setenv("KUNLUN_FLASH_DECODING_WORK","chunk",1);bool compare_layout=getenv("KUNLUN_FLASH_DECODING_COMPARE_LAYOUT");')
s=s.replace('compare_layout?"on":"off"','(compare_layout||compare_work)?"on":"off"').replace('if(compare_layout)setenv("KUNLUN_FLASH_DECODING_LAYOUT","head",1);};','if(compare_layout)setenv("KUNLUN_FLASH_DECODING_LAYOUT","head",1);if(compare_work)setenv("KUNLUN_FLASH_DECODING_WORK",work.c_str(),1);};');p.write_text(s)
p=P/'flash-decoding-bench.cpp';s=p.read_text();old='if(mode!="on"&&mode!="off"&&mode!="auto")return 2;'
s=s.replace(old,'unsetenv("KUNLUN_FLASH_DECODING_WORK");if(mode=="head"){setenv("KUNLUN_FLASH_DECODING_WORK","head",1);mode="on";}'+old);p.write_text(s)
print('Added task-order-only head affinity variants; default remains chunk mapping')

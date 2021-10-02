import os 
import shutil
import time
import sys
import argparse

########## first step: prepare results
import json

mode = 0 # 0 segm  1 kes

if mode == 0:
	with open("bo.json", 'r') as f:
		data = json.load(f)
		with open('all_detcors.txt', 'w') as f2:
				for ix in range(len(data)):
						print('Processing: '+str(ix))
						if data[ix]['score'] > 0.1:
								outstr = '{}: {},{},{},{},{},{},{},{},{}\n'.format(data[ix]['image_id'], int(data[ix]['seg_rorect'][0]),\
														int(data[ix]['seg_rorect'][1]), int(data[ix]['seg_rorect'][2]), int(data[ix]['seg_rorect'][3]),\
														int(data[ix]['seg_rorect'][4]), int(data[ix]['seg_rorect'][5]), int(data[ix]['seg_rorect'][6]),\
														int(data[ix]['seg_rorect'][7]), \
															round(data[ix]['score'], 3))
								f2.writelines(outstr)
				f2.close()
else:
	raise Exception('This is error.')


########## second step: evaluate results
dirn =  "mb_ch4_results"
eval_dir = "./"
lsc = list(range(20,90,1))
# lsc = [0.65]

f = open("evaluation_results.txt",'w')

fres = open('all_detcors.txt', 'r').readlines()

for isc in lsc:
	print('Evaluating cf threshold 0.{}:'.format(str(isc)))
	sys.stdout.flush()
		
	if os.path.exists("mb_ch4.zip"):
		os.remove("mb_ch4.zip")
	if os.path.exists("mb_ch4_results"):
		shutil.rmtree("mb_ch4_results/")

	if not os.path.isdir(dirn):
		os.mkdir(dirn)

	for line in fres:
		line = line.strip()
		s = line.split(': ')
		filename = '{:07d}.txt'.format(int(s[0]))
		outName = os.path.join(dirn, filename)
		with open(outName, 'a') as fout:
			score = s[1].split(',')[-1].strip()
			if float(score)<isc/100.:
				continue
			cors = ','.join(e for e in s[1].split(',')[:-1])
			fout.writelines(cors+'\n')
			
	os.chdir("mb_ch4_results/")
	os.popen("zip -r ../mb_ch4.zip ./")
	os.chdir("../")
	cmd = "python "+eval_dir+"script.py -g=ch4_gt.zip -s=mb_ch4.zip"
	output = os.popen(cmd).read()
	f.writelines("========= 0."+str(isc)+":\n")
	lout = output.split('\n')
	print("output is:",output)
	f.writelines(lout[1]+'\n')
	f.writelines(lout[2]+'\n')
	
	if os.path.exists("mb_ch4.zip"):
		os.remove("mb_ch4.zip")
	if os.path.exists("mb_ch4_results"):
		shutil.rmtree("mb_ch4_results/")

	f.flush()
	# time.sleep(0.01)

f.close()

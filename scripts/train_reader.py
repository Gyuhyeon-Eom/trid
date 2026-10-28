import argparse
from trid.reader import train_reader
p=argparse.ArgumentParser()
p.add_argument('--data',required=True)
p.add_argument('--init-run',required=True)
p.add_argument('--output',required=True)
p.add_argument('--steps',type=int,default=1400)
a=p.parse_args()
train_reader(a.data,a.init_run,a.output,a.steps)

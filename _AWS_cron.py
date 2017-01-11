import boto
from boto.exception import S3ResponseError
import os
import sys
import time
import threading as th

# Not working yet

'''
#from datetime import date, time, timedelta
import datetime as dt

date_s = dt.combine(dt.date.today(), dt.time(23, 55)) + dt.timedelta(minutes=30)
print date_s.dt.time()
'''

dir_path = 'saved_network/'
bucket_name = 'neural_network_backup'


# ec2 = boto3.resource('ec2')
s3_conn = boto.connect_s3()

#'saved_network/' + 'net-epoch' + str(iteration) + '.pkl'

try:
    bucket = s3_conn.get_bucket(bucket_name, validate=True)
except S3ResponseError:
    bucket = s3_conn.create_bucket(bucket_name)  #location=boto.s3.connection.Location.DEFAULT



def close():
	print 'Closing...'
	print th.current_thread()
	# sys.exit(0)
	th.interrupt_main()

def upload(dir_file):
	local_file_modtime = os.path.getmtime(dir_file)
	# s3_file_modtime = os.path.getmtime()
	if local_file_modtime > s3_file_modtime:
		key = boto.s3.key.Key(bucket, 'dir_file.zip')
		with open('some_file.zip') as f:
			key.send_file(f)

def traverse():
	global dir_path
	for dir_file in next(os.walk(dir_path))[2]:
		print dir_file
		upload(dir_file)
	t.cancel()


t = th.Timer(10.0, traverse)

if __name__ == '__main__':
	global t

	print '---AWS cron for neural networks---'
	# print 'Set time (countdown / precise) [c/p]: '
	# time_mode = raw_input()
	

	#t = datetime(2017, 01, 11, now.hour, now.minute, now.second)
	#timeout = t + timedelta(minutes=3)
	#print timeout.time()

	while True:
		# print '>> '
		# cmd = raw_input()
		# if cmd == 'time':
		# 	print ''
		t.start()


	'''if time_mode == 'c':
		print 'Set time counter [HH:MM]: '
		time_counter = raw_input()
		time.strptime(time_counter, "%H:%M")
	elif time_mode == 'p':
		print 'Set precise time []: '

	else:
		print '!!! Invalid option selected'
		print 'Closing ...\n'
		sys.exit()'''

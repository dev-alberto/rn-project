import re
import csv
import json
import sys

from datetime import datetime

#datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')


i = 0
max = 12300

pattern = 'Time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\sTotal moves made in a game: (\d{3,4})\s{3}Total Score: (\S{3})'
prog = re.compile(pattern)

with open('log_stuff.txt') as f, open('export.csv', 'w') as csvfile, open('export.json', 'w') as jsonfile, open('export_plot.json', 'w') as plotfile:
	# my_lines = f.read().split("\n")
	serializer = csv.writer(csvfile, delimiter=',', 
				quotechar='|', quoting=csv.QUOTE_MINIMAL)

	serializer.writerow(['\"Time\"', '\"Moves\"', '\"Score\"'])
	big_json = []
	big_json_plot = {}

	big_json_plot['x'] = []
	big_json_plot['y'] = []

	for line in f:
		if i < max:
			print(line)
			line_array = []
			line_dict = {}

			result = prog.match(line)
			print('Ze first result:', result.group(1))
			print('Ze second result:', result.group(2))
			print('Ze third result:', result.group(3))

			date_raw = str(result.group(1)) #shallow copy
			#2017-01-08 23:56:47
			date = datetime.strptime(result.group(1), '%Y-%m-%d %H:%M:%S')

			line_array.append(date.strftime('%H:%M:%S'))
			line_array.append(result.group(2))
			line_array.append(result.group(3))

			line_dict['Time'] = date.strftime('%H:%M:%S')
			line_dict['Moves'] = result.group(2)
			line_dict['Score'] = result.group(3)
			big_json.append(line_dict)

			big_json_plot['x'].append(date.strftime('%H:%M:%S'))
			# big_json_plot['y'].append(result.group(2)) # Moves
			big_json_plot['y'].append(result.group(3)) # Score


			serializer.writerow(line_array)

		else:
			print('\nReached the threshold, closing ...')
			break
		i += 1

	if i < max:
		print('\nFinished file before reaching the threshold, closing ...')

	#jsonfile.write( json.dumps(big_json, sort_keys=True, indent=4) )
	json.dump(big_json, jsonfile, sort_keys=True, indent=4)
	json.dump(big_json_plot, plotfile, sort_keys=True, indent=4)

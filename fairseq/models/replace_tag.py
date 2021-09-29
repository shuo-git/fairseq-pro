import sys

my_tag = "<{}>".format(sys.argv[1])
res_tag = "<{}2>".format(sys.argv[1])

for line in sys.stdin:
	words = line.split()
	assert words[0] == my_tag
	line2 = ' '.join(words[1:])
	res_line = my_tag + ' ' + line2.replace(my_tag, res_tag)
	print(res_line)

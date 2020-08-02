import time
print('Starting')
print("Saving")
# file = open("results.txt","w")
# time.sleep(60)
file = open('logs.txt', 'w')
print('hello world', file=file)
file.close()

# file.write('testing script')
# file.close()
print('done')
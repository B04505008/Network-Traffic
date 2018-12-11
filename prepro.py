
import os
import time
from scapy.all import *
#from pcapng.scanner import FileScanner
import numpy as np
import os
import multiprocessing as mp
import pickle as pk
from utils import *



with open('objs/fileName2Application.pickle', 'rb') as f:
    dict_name2label = pk.load(f)


with open('objs/fileName2Characterization.pickle', 'rb') as f:
    dict_name2class = pk.load(f)


def pkts2X(pkts,filename):
    X = []

    tail = os.path.split(filename)[1]
    cl = dict_name2class[tail]
    #lens = []
    for p in pkts:
        #===================================
        # step 1 : remove Ether Header
        #===================================
        r = raw(p)[34:]#14 to 54
        r = np.frombuffer(r, dtype = np.uint8)
        #p.show()
        #===================================
        # step 2 : pad 0 to UDP Header
        # it seems that we need to do nothing this step
        # I found some length of raw data is larger than 1500
        # remove them.
        #===================================
        if (TCP in p or UDP in p):
            """
            if UDP in p:
                # todo : padding 0 to 
                print ('UDP', r[:20])
                print(p[IP].src, p[IP].dst)
            else :
                print('TCP', r[:20])
                print(p[IP].src, p[IP].dst)
            """
            if (IP not in p or (p[IP].src.find('224.0.0') != -1 ) or (p[IP].dst.find('224.0.0') != -1 )or (p[IP].src.find('255.255.250') != -1 ) or (p[IP].dst.find('255.255.250') != -1 )):
                pass
            elif ((p[IP].src.find('255.255.255') != -1 ) or (p[IP].dst.find('255.255.255') != -1 ) or DNS in p or ICMP in p ): 
                pass
            elif ( NBNSRequest in p or NBNSWackResponse in p or NBNSQueryRequest in p or NBNSQueryResponse in p or NBNSNodeStatusResponse in p or NBNSNodeStatusResponseService in p):
                pass
            elif(NetBIOS_DS in p or NBTDatagram in p or NBTSession in p):
                pass
            ########file transfer should not use udp###################
            elif ((UDP in p) and ( cl == 'Torrent' or cl == 'VPN: Torrent' or cl == 'File Transfer' or cl == 'VPN: File Transfer')):
                pass
            ####################################################
            else:
                r = r[:1500]
                X.append(r)
               
        else:
            pass
    return X


def get_data_by_file(filename):
    pkts = rdpcap(filename)
    X = pkts2X(pkts,filename)
    # save X to npy and delete the original pcap (it's too large).
    return X 

def task(filename):
    global dict_name2label
    global counter
    head, tail = os.path.split(filename)
    cond1 = os.path.isfile(os.path.join('data1d', tail+'.pickle'))
    cond2 = os.path.isfile(os.path.join('data1d', tail+'_class.pickle'))
    
    
    if (cond1 and cond2):
        with lock:
            counter.value += 1        
        print('[{}] {}'.format(counter, filename))
        return '#ALREADY#'
    X = get_data_by_file(filename)
    if (not cond1):
        y = [dict_name2label[tail]] * len(X)
        with open(os.path.join('data1d', tail+'.pickle'), 'wb') as f:
            pk.dump((X, y), f)
    if (not cond2):
        y2 = [dict_name2class[tail]] * len(X)
        with open(os.path.join('data1d', tail+'_class.pickle'), 'wb') as f:
            pk.dump(y2, f)
            
    
    with lock:
        counter.value += 1
    print('[{}] {}'.format(counter, filename))
    return 'Done'


#=========================================
# mp init
#=========================================
lock = mp.Lock()
counter = mp.Value('i', 0)
cpus = mp.cpu_count()//2
pool = mp.Pool(processes=cpus)



todo_list = gen_todo_list('./pcaps')

#todo_list = todo_list[:3]

total_number = len(todo_list)

done_list = []
    
res = pool.map(task, todo_list)

print(len(res))



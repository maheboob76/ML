from bs4 import BeautifulSoup
#import urllib2
import requests
import sys
sys.executable

base_url = 'http://results.eci.gov.in/pc/en/constituencywise/'
DATA_DIR = 'D:/Amaan/Code/GitRepos/ML/India Election 2019/polling_data/'

def create_state_dict(filename):
    soup = BeautifulSoup( open(filename) )
    
    raw_states = soup.find_all('option')
    states = {}
    for ops in raw_states:
        print(ops.get('value'), ops.get_text())
        states[ops.get('value')] = ops.get_text()
        
    return states


#states = create_state_dict('state_list.html')

def parse_const_list(data):
    
    #print('----> ', data)
    '''dropping last semi colon '''
    data = data[:-1]
    const_list = {}
    
    raw_const = data.split(';')
    for const in raw_const:
        const_id, const_name = const.split(',')
        const_list[const_id] = const_name
    
    
    
    return const_list
    
    

def create_statewise_const_dict(filename):
    soup = BeautifulSoup( open(filename) )
    
    print(soup)
    raw_states = soup.find_all('input')
    states = {}
    for ops in raw_states:
        print(ops.get('id'), ops.get('value'))
        states[ops.get('id')] = parse_const_list(ops.get('value'))
        
    return states
    
    
def create_const_dict(filename):
    soup = BeautifulSoup( open(filename) )
    
    print(soup)
    raw_states = soup.find_all('input')
    states = {}
    for ops in raw_states:
        print(ops.get('id'), ops.get('value'))
        states[ops.get('id')] = ops.get('value')
        
    return states

#state_const_list = create_const_list('main pgae.html')
#state_const_list = create_statewise_const_dict('main pgae.html')

def create_download_links(state_const_list):
       
    download_list =[]
    
    for state in state_const_list:
        #print('state list: ', state)
        const_list = parse_const_list( state_const_list[state])
        print(const_list)
        
        for const in const_list:
            print(const, ':', const_list[const])
            link = base_url + 'Constituencywise' + state + const + '.htm?' + 'ac=' + const
            print(link)
            download_list.append(link)
           
            
    return download_list
   
#ll = create_download_links(state_const_list)

def get_filename_from_url(url):
    const = url.split('/')[-1]
    const = const.split('?')[0]
    #print('filenmae before returning: ', const)
    
    return const
    
def download_n_save(url , target_dir):
    print('URL: ', url)
    r = requests.get(url)
    #get_const_from_url(url)
    
    filename = target_dir + get_filename_from_url(url)
    out_file = open(filename, 'w')
           
    out_file.write(str(r.content))
    out_file.close()
    # or without a session: r = requests.get(url)
    #print(r.content)

def download_all():
    state_const_list = create_const_dict('main pgae.html')
    all_links = create_download_links(state_const_list)
    for l in all_links:
        download_n_save(l, DATA_DIR)

from bs4 import BeautifulSoup
import pandas as pd
from os import listdir
import create_links

DATA_DIR = 'D:/Amaan/Code/GitRepos/ML/India Election 2019/polling_data/'

''' Extracts and returns state and constituency from filename '''

states = create_links.create_state_dict('state_list.html')
state_const_list = create_links.create_statewise_const_dict('main pgae.html')

def extract_state_const(filename):
    print('FILE:', filename)
    state = ''
    const = ''
    
    temp = filename.split('.')[0] #drop extension
    state_code = temp[16:19]
    const_code = temp[19:]
    
    state = states[state_code]
    const = state_const_list[state_code][const_code]
    
    return state, const
   
#s, c = extract_state_const('ConstituencywiseS0424.htm')
    
def parse_data(filename):
    
    print('filename: ', filename)
    column_headers = ['State', 'Constituency', 'Candidate', 'Party', 'EVM_Votes', 'Migrant_Votes', 'Postal_Votes', 'Total_Votes', 'Perct_Votes']
    state, const = extract_state_const(filename.split('/')[-1])   
    soup = BeautifulSoup( open(filename), 'html.parser' )
    main_list = []
    table = soup.find_all('table')[10]
    
    rows = table.find_all('tr')[3:]
    for r in rows:
        #print('row : ')
        columns = r.find_all('td')[1:]
        
        row_list = []
        for c in columns:
            print('columns:' , c.get_text())
            row_list.append(c.get_text())
            
            ''' if its 6 columns then add extra column in position in 4th column; e.g. index 3 '''
        print('ROW before adding migrant column: ' , len(row_list))
        if len(row_list) == 6:
            row_list.insert(3,-1)
            
        row_list.insert(0, const) # insert for each row
        row_list.insert(0, state) # insert for each row
        
        
        main_list.append(row_list)
        
     
    
    
    
    #print(  rows)
    
    #print(main_list)
    df = pd.DataFrame(main_list, columns=column_headers)
    df["EVM_Votes"] = df['EVM_Votes'].astype('int')
    df["Migrant_Votes"] = df['Migrant_Votes'].astype('int')
    df["Postal_Votes"] = df['Postal_Votes'].astype('int')
    df["Total_Votes"] = df['Total_Votes'].astype('int')
    #df["Perct_Votes"] = df['Perct_Votes'].astype('float32')
    
    
    
    return  df
    
#t = parse_data(DATA_DIR+ 'ConstituencywiseS036.htm')
#df["Customer Number"] = df['Customer Number'].astype('int')

def prepare_excel(source_dir):
    files = listdir(DATA_DIR)
   
    
    main_list = pd.DataFrame()
    
    for f in files:
        print(f)
        main_list = main_list.append(parse_data(source_dir + f))
        
    return main_list
    

''' top party by percentage of votes '''
def calculate_top_party(df):
    
    df_party = df.groupby(['Party'])['Total_Votes'].sum()
    df.reset_index(inplace = True) 
    
    df_party.sort_values(ascending = False)[:10].plot(kind='bar')
    df_top_10 = df_party.sort_values(ascending = False)[:10]
    df_top_10 = df_top_10/10000000
    df_top_10 = 100 * df_top_10/total_votes
    
    return df_top_10


main_df = prepare_excel(DATA_DIR)

''' remove total '''
df = main_df[ main_df.Candidate != 'Total' ]

total_votes = df['Total_Votes'].sum() / 10000000

#df_const_wise = df.groupby(['State', 'Constituency'])['Total_Votes'].nlargest(2)
df_const_wise = df.groupby(['State','Party'])['Total_Votes'].sum()
df_const_wise = pd.DataFrame(df_const_wise)
df_const_wise.reset_index(inplace=True)


pivot_df = df_const_wise.pivot(index='State', columns='Party', values='Total_Votes')
pivot_df

pivot_df.loc[:,['Bharatiya Janata Party','Indian National Congress']].plot.bar(stacked=False,  figsize=(20,10))



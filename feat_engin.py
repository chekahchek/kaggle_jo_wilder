import numpy as np
import pandas as pd

def get_general_features(df):
    dfs = []
    
    # 1 - Number of unique text
    tmp = df.groupby('session_id')['text'].nunique()
    tmp.name = 'unique_text'
    dfs.append(tmp)

    # 2 - Total elapsed time
    tmp = df.groupby('session_id')['elapsed_time'].sum()
    tmp.name = 'total_elapsed_time'
    dfs.append(tmp)

    # 3 - Total hover duration
    tmp = df.groupby('session_id')['hover_duration'].sum()
    tmp.name = 'total_hover_duration'
    dfs.append(tmp)

    # 4 - Number of times notebook open
    tmp = df.loc[df['event_name']  == 'notebook_click', :].groupby('session_id')['event_name'].count()
    tmp.name = 'total_notebook_click'
    dfs.append(tmp)

    # 5 - Time spent for each event_name
    EVENT_NAMES = ['navigate_click','person_click','cutscene_click','object_click', 'map_hover','notification_click','map_click','observation_click']
    for event_name in EVENT_NAMES:
        tmp = df.loc[df['event_name'] == event_name, :].groupby('session_id')['elapsed_time'].sum()
        tmp.name = event_name + '_time'
        dfs.append(tmp)

    train = pd.concat(dfs,axis=1).reset_index()
    return train



def get_answer_time_1(df, train=True):
    df['relevant'] = (df['event_name'] == 'checkpoint')
    df['relevant2'] = df['relevant'].shift(-1)
    df['keep'] = df['relevant'] | df['relevant2']
    
    if train == False and sum(df['keep']) < 2:
        df = df.iloc[-2:, :]
    else:
        df = df.loc[df['keep'], :]

    df['answer_time_1'] = df.groupby('session_id')['elapsed_time'].shift(1)
    df['answer_time_1'] = df['elapsed_time'] - df['answer_time_1']
    df = df.loc[df['answer_time_1'].notna(), :]
    
    if train:
        df = df.groupby('session_id')[['session_id', 'answer_time_1']].head(1).reset_index(drop=True)
    else:
        df = df[['answer_time_1']].head(1).reset_index(drop=True)
    return df


    
def get_event_details(df, s_level, e_level, s_text_fqid=None, e_text_fqid=None, s_room_fqid=None,  e_room_fqid=None, train=True, rm_count=False, num_id=-1):
    
    if train:
        print(f"Num of sess_id at the start for {num_id}: {df['session_id'].nunique()}")
        if s_room_fqid == None or e_room_fqid == None:
            df['start_index'] = (df['level'] == s_level) & (df['text_fqid'] == s_text_fqid)
            df['end_index'] = (df['level'] == e_level) & (df['text_fqid'] == e_text_fqid)

        elif s_text_fqid == None or e_text_fqid == None:
            df['start_index'] = (df['level'] == s_level) & (df['room_fqid'] == s_room_fqid)
            df['end_index'] = (df['level'] == e_level) & (df['room_fqid'] == e_room_fqid)

        else:
            raise Exception("Provide either text_fqid or room_fqid")

        # To get rid of rows where either start or end text is missing
        df['start_present'] = (df.groupby('session_id')['start_index'].transform(lambda x: x.sum()) >= 1).astype(int)
        df['end_present'] =  (df.groupby('session_id')['end_index'].transform(lambda x: x.sum()) >= 1).astype(int)
        df['keep'] = df['start_present'] + df['end_present'] == 2
        df = df.loc[df['keep'], :]

        # Keep rows that fall within start and end index
        df['start_index'] = df['start_index'].mask(df['start_index'], df['index'])
        df['end_index'] = df['end_index'].mask(df['end_index'], df['index'])

        df.loc[df['start_index'] == False, 'start_index'] = np.NaN
        df.loc[df['end_index'] == False, 'end_index'] = np.NaN
        df['keep'] = df.groupby('session_id').apply(lambda x: (x['index'] >= x['start_index'].min()) & (x['index'] <= x['end_index'].max())).values
        df = df.loc[df['keep'], ['session_id', 'elapsed_time', 'event_name', 'room_fqid']]
        
        
        if rm_count:
            out = df.groupby('session_id').agg(event_counts=('event_name', 'count'), room_unique_count=('room_fqid', 'nunique'))
            out = out.rename(columns={'event_counts' : f'event_counts_{str(num_id)}', 
                                      'room_unique_count' : f'room_unique_count_{str(num_id)}'
                                     })
        else:
            out = df.groupby('session_id').agg(event_counts=('event_name', 'count'))
            out = out.rename(columns={'event_counts' : f'event_counts_{str(num_id)}'})
        
        out[f'time_taken_{str(num_id)}'] = df.groupby('session_id').nth(-1)['elapsed_time'] - df.groupby('session_id').nth(0)['elapsed_time']
    
        print(f"Num of sess_id at the end for {num_id}: {len(out)}")

        return out.reset_index()
    
    # For inference
    else:
        if s_room_fqid == None or e_room_fqid == None:
            column = 'text_fqid'
            start_condition = s_text_fqid
            end_condition = e_text_fqid
        elif s_text_fqid == None or e_text_fqid == None:
            column = 'room_fqid'
            start_condition = s_room_fqid
            end_condition = e_room_fqid
        elif s_room_fqid != None and e_room_fqid != None and s_text_fqid != None and e_text_fqid != None:
            column = None
        else:
            raise Exception("Provide either text_fqid or room_fqid")
        
        if column is not None:
            s_cri = (df['level'] == s_level) & (df[column] == start_condition)
            e_cri = (df['level'] == e_level) & (df[column] == end_condition)

            if len(df.loc[s_cri, 'index']) != 0 and len(df.loc[e_cri, 'index']) != 0:
                s_index = df.loc[s_cri, 'index'].idxmin()
                e_index = df.loc[e_cri, 'index'].idxmax()
                
            elif len(df.loc[df['level'] == s_level, 'index']) != 0 and len(df.loc[df['level'] == e_level, 'index']) != 0:
                s_index = df.loc[df['level'] == s_level, 'index'].idxmin()
                e_index = df.loc[df['level'] == e_level, 'index'].idxmax()
            else:
                s_index = df.index[0]
                e_index = df.index[-1]

            
        else:
            s_cri_1 = (df['level'] == s_level) & (df['text_fqid'] == s_text_fqid)
            e_cri_1 = (df['level'] == e_level) & (df['text_fqid'] == e_text_fqid)
            
            s_cri_2 = (df['level'] == s_level) & (df['room_fqid'] == s_room_fqid)
            e_cri_2 = (df['level'] == e_level) & (df['room_fqid'] == e_room_fqid)
            
            if len(df.loc[s_cri_1, 'index']) != 0 and len(df.loc[e_cri_1, 'index']) != 0:
                s_index = df.loc[s_cri_1, 'index'].idxmin()
                e_index = df.loc[e_cri_1, 'index'].idxmax()
                
            elif len(df.loc[s_cri_2, 'index']) != 0 and len(df.loc[e_cri_2, 'index']) != 0:
                s_index = df.loc[s_cri_2, 'index'].idxmin()
                e_index = df.loc[e_cri_2, 'index'].idxmax()
                
            elif len(df.loc[df['level'] == s_level, 'index']) != 0 and len(df.loc[df['level'] == e_level, 'index']) != 0:
                s_index = df.loc[df['level'] == s_level, 'index'].idxmin()
                e_index = df.loc[df['level'] == e_level, 'index'].idxmax()
            else:
                s_index = df.index[0]
                e_index = df.index[-1]


        df = df.iloc[s_index:e_index].reset_index(drop=True)
        time_taken = df['elapsed_time'].iloc[-1] - df['elapsed_time'].iloc[0]
        event_count = df['event_name'].count()
        
        if rm_count:
            room_uniq_count = df['room_fqid'].nunique()
            out = pd.DataFrame({f'time_taken_{str(num_id)}' : time_taken, 
                                f'event_counts_{str(num_id)}' : event_count,
                                f'room_unique_count_{str(num_id)}' : room_uniq_count},
                                index=[0])
                             
        else:
            out = pd.DataFrame({f'time_taken_{str(num_id)}' : time_taken, 
                                f'event_counts_{str(num_id)}' : event_count}, 
                                index=[0])
            
        return out
import numpy as np
import pandas as pd

def get_general_features(df, stage, train=True):
    dfs = []
    
    # 1 - Number of unique text
    tmp = df.groupby('session_id')['text'].nunique()
    tmp.name = 'unique_text'
    dfs.append(tmp)

    # 2 - Total elapsed time
    tmp = df.groupby('session_id')['elapsed_time'].max()
    tmp.name = 'total_elapsed_time'
    dfs.append(tmp)
    
    # 3 - Mean elapsed time 
    tmp = df.groupby('session_id')['elapsed_time'].mean()
    tmp.name = 'mean_elapsed_time'
    dfs.append(tmp)
    
    # 4 - Std Dev elapsed time 
    tmp = df.groupby('session_id')['elapsed_time'].std()
    tmp.name = 'stddev_elapsed_time'
    dfs.append(tmp)
    
    # 5 - Average Time per action
    tmp = df.groupby('session_id')['elapsed_time'].max() / df.groupby('session_id').size()
    tmp.name = 'time_per_action'
    dfs.append(tmp)
    
    # 6 - Total hover duration
    tmp = df.groupby('session_id')['hover_duration'].sum()
    tmp.name = 'total_hover_duration'
    dfs.append(tmp)

    # 7 - Number of times notebook open
    tmp = df.loc[df['event_name']  == 'notebook_click', :].groupby('session_id')['event_name'].count()
    tmp.name = 'total_notebook_click'
    dfs.append(tmp)

    # 8 - Time spent for each event_name
    EVENT_NAMES = ['navigate_click','person_click','cutscene_click','object_click', 'map_hover','notification_click','map_click','observation_click']
    for event_name in EVENT_NAMES:
        tmp = df.loc[df['event_name'] == event_name, :].groupby('session_id')['action_time'].sum()
        tmp.name = event_name + '_time'
        dfs.append(tmp)

    _train = pd.concat(dfs,axis=1).reset_index()
    
    # 9 - Time per level
    tmp = (df.groupby(['session_id', 'level'])[['elapsed_time']].max() - df.groupby(['session_id', 'level'])[['elapsed_time']].min())
    tmp = tmp.reset_index().pivot_table(index='session_id', values='elapsed_time', columns='level').reset_index()
    col_names = {col: f"lvl_{col}_time" for col in tmp.columns if col != 'session_id'}
    tmp = tmp.rename(columns=col_names)
    
    if train == False:
        if stage == 1 and len(tmp.columns) != 6:
            for lvl in range(0, 5):
                lvl_col = f"lvl_{lvl}_time"
                if lvl_col not in tmp.columns:
                    tmp[lvl_col] = 0 
            tmp = tmp[['session_id'] + [f"lvl_{i}_time" for i in range(0,5)]]
            
        elif stage == 2 and len(tmp.columns) != 9:
            for lvl in range(5, 13):
                lvl_col = f"lvl_{lvl}_time"
                if lvl_col not in tmp.columns:
                    tmp[lvl_col] = 0
            tmp = tmp[['session_id'] + [f"lvl_{i}_time" for i in range(5,13)]]

        elif stage == 3 and len(tmp.columns) != 11:
            for lvl in range(13, 23):
                lvl_col = f"lvl_{lvl}_time"
                if lvl_col not in tmp.columns:
                    tmp[lvl_col] = 0
            tmp = tmp[['session_id'] + [f"lvl_{i}_time" for i in range(13,23)]]
    
    _train = pd.merge(left=_train, right=tmp, on='session_id', how='left')
    return _train



def get_answer_time_1(df, train=True):
    df['relevant'] = (df['event_name'] == 'checkpoint')
    df['relevant2'] = df['relevant'].shift(-1)
    df['keep'] = df['relevant'] | df['relevant2']
    
    if train == False and sum(df['keep']) < 2:
        df = df.iloc[-2:, :]
    else:
        df = df.loc[df['keep'], :]

    if train:
        df = df.groupby('session_id')[['session_id', 'action_time']].head(1).reset_index(drop=True)
    else:
        df = df[['action_time']].head(1).reset_index(drop=True)
        
    df['action_time'] = df['action_time'].clip(upper=50000) 
    df = df.rename(columns={'action_time' : 'answer_time_1'})
    return df


def get_answer_time_2(stage3_df, stage2_path=None, retained_features=None, train=True):
    if train:
        data = pd.read_csv(stage2_path)
        stage2_answertime = data.groupby('session_id').nth(-1)[['elapsed_time']]
        stage3_answertime = stage3_df.groupby('session_id').nth(0)[['elapsed_time']]

        out = pd.merge(stage3_answertime, stage2_answertime, left_index=True, right_index=True, how='left')
        out['answer_time_2'] = out['elapsed_time_x'] - out['elapsed_time_y']
        out['answer_time_2'] = out['answer_time_2'].clip(lower=0)
        out = out['answer_time_2'].reset_index()
        
    else:
        sess = stage3_df['session_id'].iloc[0]
        stage2_answertime = retained_features[sess]['stage2_answertime']
        stage3_answertime = stage3_df['elapsed_time'].iloc[0]
        answer_time_2 = stage3_answertime - stage2_answertime
        if answer_time_2 < 0: answer_time_2 = 0
        out = pd.DataFrame({'answer_time_2': answer_time_2}, index=[0])
        
    return out
    
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
            out = pd.DataFrame({f'event_counts_{str(num_id)}' : event_count,
                                f'room_unique_count_{str(num_id)}' : room_uniq_count,
                                f'time_taken_{str(num_id)}' : time_taken}, 
                                index=[0])
                             
        else:
            out = pd.DataFrame({f'event_counts_{str(num_id)}' : event_count,
                                f'time_taken_{str(num_id)}' : time_taken}, 
                                index=[0])
            
        return out
    

def arrow_click(df, substring, name):
    """
    For events where user has to click through multiple pages before selecting the correct one. 
    cri = Filtering for situation where the user is navigating through all the different pages. Doesm't incldue the navigate click to take them into the scren,
    closing the event screen but include the bingo and the amount of hovering done
    cri_time = Filtering for situation where user must first navigate into the screen all the way till bingo (inclusive). This is done to find out the 
               amount of time on the event itself.
    
    Stage 2 - Business card : substring = 'businesscards', name = 'bizcard'
    Stage 2 - Microfiche : substring = 'reader', name = 'microfiche'
    Stage 2 - Stacks : substring = 'journals.pic', name = 'stackjournals'
    Stage 3 - Ecology flag in the microfiche : substring = 'reader_flag', name = 'readerflag'
    Stage 3 - Activist in front of flag: substring = 'journals_flag', name = 'activistflag'
    """
    
    df['substring'] = df['fqid'].str.contains(substring)
    cri = (df['substring']) & (df['event_name'] != 'navigate_click') & (df['name'] != 'close')
    cri_time = (df['substring']) & (df['name'] != 'close')
    
    df[name] = cri.astype(int)
    tmp = df.groupby('session_id')[[name]].sum()
    
    tmp2 = df.loc[cri_time, :].groupby('session_id')[['elapsed_time']].max() - df.loc[cri_time, :].groupby('session_id')[['elapsed_time']].min()
    out = pd.merge(tmp, tmp2, left_index=True, right_index=True, how='left').reset_index().rename(columns={'elapsed_time' : f'{name}_elapsed_time'})
    return out


def point_click(df, fqid, fqid_bingo, name):
    """
    For events where user has to click on the correct option in the entire page. 
    cri = Filtering for situation where the user is clicking on the correct option. Doesn't include the navigate click to take them into the screen,
          closing the event screen, or the bingo but include the amount of hovering done
    cri_time = Filtering for situation where user must first navigate into the screen all the way till bingo (inclusive). This is done to find out the 
               amount of time on the event itself.
    
    Stage 1 - Plaque : fqid = 'plaque', bingo_fqid = 'plaque.face.date', name = 'plaque_clicks'
    Stage 2 - Logbooks : fqid = 'logbook', bingo_fqid = 'logbook.page.bingo', name = 'logbook_clicks'
    Stage 3 - Directory : fqid = 'directory', bingo_fqid = 'directory.closeup.archivist', name = 'directory_specs'
    """
    cri = (df['fqid'] == fqid) & (df['event_name'] != 'navigate_click') & (df['name'] != 'close')
    cri_time = ((df['fqid'] == fqid) | (df['fqid'] == fqid_bingo)) & (df['name'] != 'close')
    
    df[name] = (cri).astype(int)
    tmp = df.groupby('session_id')[[name]].sum()
    
    tmp2 = df.loc[cri_time, :].groupby('session_id')[['elapsed_time']].max() - df.loc[cri_time, :].groupby('session_id')[['elapsed_time']].min()
    out = pd.merge(tmp, tmp2, left_index=True, right_index=True, how='left').reset_index().rename(columns={'elapsed_time' : f'{name}_elapsed_time'})
    
    return out
    
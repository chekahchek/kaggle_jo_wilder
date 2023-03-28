import numpy as np
import pandas as pd

def get_general_features(df, stage, train=True):
    dfs = []
    
    # 1 - Number of unique text
    tmp = df.groupby('session_id')['text'].nunique()
    tmp.name = 'unique_text'
    dfs.append(tmp)

#     # 2 - Total elapsed time
#     tmp = df.groupby('session_id')['elapsed_time'].max()
#     tmp.name = 'total_elapsed_time'
#     dfs.append(tmp)
    
#     # 3 - Mean elapsed time 
#     tmp = df.groupby('session_id')['elapsed_time'].mean()
#     tmp.name = 'mean_elapsed_time'
#     dfs.append(tmp)
    
#     # 4 - Std Dev elapsed time 
#     tmp = df.groupby('session_id')['elapsed_time'].std()
#     tmp.name = 'stddev_elapsed_time'
#     dfs.append(tmp)
    
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
    # EVENT_NAMES = ['navigate_click','person_click','cutscene_click','object_click', 'map_hover','notification_click','map_click','observation_click']
    EVENT_NAMES = ['navigate_click', 'person_click','cutscene_click','object_click', 'map_hover','observation_click']
    for event_name in EVENT_NAMES:
        tmp = df.loc[df['event_name'] == event_name, :].groupby('session_id')['action_time'].sum()
        tmp.name = event_name + '_time_sum'
        dfs.append(tmp)
        
        tmp = df.loc[df['event_name'] == event_name, :].groupby('session_id')['action_time'].mean()
        tmp.name = event_name + '_time_mean'
        dfs.append(tmp)
        
        tmp = df.loc[df['event_name'] == event_name, :].groupby('session_id')['action_time'].std()
        tmp.name = event_name + '_time_std'
        dfs.append(tmp)

    _train = pd.concat(dfs,axis=1).reset_index()
    
    # 9 - Time per level - Sum, Mean, Median, Std
#     tmp = df.groupby(['session_id', 'level']).agg({'action_time' : ['sum', 'mean', 'median', 'std']}).reset_index()
#     tmp.columns = tmp.columns.map(''.join)
#     tmp_pivot = tmp.pivot_table(index='session_id', columns='level', values=['action_timesum', 'action_timemean', 'action_timemedian', 'action_timestd'])
#     tmp_pivot.columns = [i[0] + '_' + str(i[1]) for i in tmp_pivot.columns]
#     tmp_pivot = tmp_pivot.reset_index()
    
#     if train == False:
#         ADD_COLUMNS = False
#         if stage == 1 and len(tmp_pivot.columns) != 21:
#             ADD_COLUMNS = True
#             level_range = range(0,5)
#         elif stage == 2 and len(tmp_pivot.columns) != 33:
#             ADD_COLUMNS = True
#             level_range = range(5,13)
#         elif stage == 3 and len(tmp_pivot.columns) != 41:
#             ADD_COLUMNS = True
#             level_range = range(13,23)
            
#         if ADD_COLUMNS:
#             NEEDED_COLS = ['session_id'] + ['action_timemean_' + str(i) for i in level_range] + ['action_timemedian_' + str(i) for i in level_range] + ['action_timestd_' + str(i) for i in level_range] + ['action_timesum_' + str(i) for i in level_range]
#             missing_cols = np.array(NEEDED_COLS[1:])[~np.isin(NEEDED_COLS[1:], tmp_pivot.columns)]
#             for _col in missing_cols:
#                 tmp_pivot[_col] = 0
#             tmp_pivot = tmp_pivot[NEEDED_COLS]
        
#     _train = pd.merge(left=_train, right=tmp_pivot, on='session_id', how='left')
    
    
    #10 - Time per level and event
    COLS = []
    EVENT_AT_LEVEL_STG1 = {
        'level0_events' : ['navigate_click', 'notification_click', 'object_click'],
        'level1_events' : ['cutscene_click', 'object_click'],
        'level2_events' : ['cutscene_click', 'navigate_click', 'object_click'],
        'level3_events' : ['notification_click'],
        'level4_events' : ['navigate_click', 'map_click']
    }

    EVENT_AT_LEVEL_STG2 = {
        'level5_events' : ['map_click'],
        'level6_events' : ['cutscene_click', 'navigate_click', 'observation_click'],
        'level7_events' : ['navigate_click', 'object_click', 'person_click'],
        'level8_events' : ['object_click', 'person_click'],
        'level9_events' : ['navigate_click', 'object_click'],
        'level10_events' : ['navigate_click', 'object_click', 'person_click'],
        'level11_events' : ['map_click', 'navigate_click', 'notification_click', 'object_click'],
        'level12_events' : ['navigate_click']
    }

    EVENT_AT_LEVEL_STG3 = {
        'level13_events' : ['navigate_click'],
        'level14_events' : ['navigate_click'],
        'level15_events' : ['person_click'],
        'level16_events' : ['cutscene_click'],
        'level17_events' : ['cutscene_click', 'navigate_click'],
        'level18_events' : ['navigate_click', 'notification_click', 'object_click'],
        'level19_events' : ['navigate_click', 'notification_click', 'object_click'],
        'level20_events' : ['map_click', 'navigate_click', 'notification_click', 'object_click'],
        'level21_events' : ['map_click', 'navigate_click', 'notification_click', 'object_click'],
        'level22_events' : ['navigate_click']
    }


    tmp = df.groupby(['session_id', 'level', 'event_name']).agg({'action_time' : ['sum', 'mean']}).reset_index()
    tmp.columns = tmp.columns.map(''.join)
    tmp_pivot = tmp.pivot_table(index='session_id', columns=['level', 'event_name'], values=['action_timesum', 'action_timemean'])
    tmp_pivot.columns = [str(i[1]) + '_' + i[2] + '_' + i[0] for i in tmp_pivot.columns]
    
    if stage == 1:
        level_range = range(0,5)
        EVENT_AT_LEVEL = EVENT_AT_LEVEL_STG1
    elif stage == 2:
        level_range = range(5,13)
        EVENT_AT_LEVEL = EVENT_AT_LEVEL_STG2
    else:
        level_range = range(13,23)
        EVENT_AT_LEVEL = EVENT_AT_LEVEL_STG3
        
    for level, cols in zip(list(level_range), EVENT_AT_LEVEL.values()):
        COLS.extend([str(level) + '_' + i + '_' + 'action_timesum' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemean' for i in cols])
    
    if train == False:
        ADD_COLUMNS = False
        if stage == 1 and len(tmp_pivot.columns) != 37:
            ADD_COLUMNS = True
        elif stage == 2 and len(tmp_pivot.columns) != 65:
            ADD_COLUMNS = True
        elif stage == 3 and len(tmp_pivot.columns) != 57:
            ADD_COLUMNS = True
            
        if ADD_COLUMNS:
            missing_cols = np.array(COLS)[~np.isin(COLS, tmp_pivot.columns)]
            for _col in missing_cols:
                tmp_pivot[_col] = 0
            
    
    tmp_pivot = tmp_pivot[COLS]
    tmp_pivot = tmp_pivot.reset_index()
    
    _train = pd.merge(left=_train, right=tmp_pivot, on='session_id', how='left')
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
        # data = pd.read_csv(stage2_path, dtype={'session_id':'object', 'elapsed_time':np.int32})
        data = pd.read_parquet(stage2_path)
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


def get_datetime_feat(data):

    data["year"] = data["session_id"].apply(lambda x: int(str(x)[:2])).astype(np.uint8)
    data["month"] = data["session_id"].apply(lambda x: int(str(x)[2:4]) + 1).astype(np.uint8)
    data["weekday"] = data["session_id"].apply(lambda x: int(str(x)[4:6])).astype(np.uint8)
    data['weekend'] = ((data['weekday'] == 0) | (data['weekday'] == 1)).astype(int)
    data["hour"] = data["session_id"].apply(lambda x: int(str(x)[6:8])).astype(np.uint8)
    
    return data
    
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


def game_version(data):
    
    cri = (data['text'].str.contains("because history is boring!")) | (data['text'].str.contains("Meetings are BORING!"))
    original = cri.fillna(False).astype(int).sum()

    cri = (data['text'].str.contains("Ugh. Meetings are so boring")) | (data['text'].str.contains("Do I have to?"))
    no_humour = cri.fillna(False).astype(int).sum()

    cri = data['text'].str.contains("Yes! This cool old slip from 1916")
    no_snark = cri.fillna(False).astype(int).sum()

    cri = (data['text'].str.contains("Yes! This old slip from 1916.")) | (data['text'].str.contains("Sure!"))
    dry = cri.fillna(False).astype(int).sum()
    
    if dry > 0:
        label = 3 
    elif no_snark > 0:
        label = 2
    elif no_humour > 0:
        label = 1
    else:
        label = 0
        
    return pd.DataFrame({'version': label}, index=[0])

    
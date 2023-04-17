import numpy as np
import pandas as pd

def get_general_features(df, stage, train=True):
    dfs = []
    
    # Number of unique text
    tmp = df.groupby('session_id')['text'].nunique()
    tmp.name = 'unique_text'
    dfs.append(tmp)
    
    # Length of dataframe
    tmp = df.groupby('session_id')['index'].count()
    tmp.name = 'df_length'
    dfs.append(tmp)

    # Total elapsed time
    tmp = df.groupby('session_id')['elapsed_time'].max()
    tmp.name = 'total_elapsed_time'
    dfs.append(tmp)
    
    # Mean elapsed time 
    tmp = df.groupby('session_id')['elapsed_time'].mean()
    tmp.name = 'mean_elapsed_time'
    dfs.append(tmp)
    
    # Std Dev elapsed time 
    tmp = df.groupby('session_id')['elapsed_time'].std()
    tmp.name = 'stddev_elapsed_time'
    dfs.append(tmp)
    
    # Median elapsed time 
    tmp = df.groupby('session_id')['elapsed_time'].median()
    tmp.name = 'median_elapsed_time'
    dfs.append(tmp)
    
    # 5 - Average Time per action
    tmp = df.groupby('session_id')['elapsed_time'].max() / df.groupby('session_id').size()
    tmp.name = 'time_per_action'
    dfs.append(tmp)
    
    # Hover duration
    tmp = df.loc[df['event_name']  == 'object_hover', :].groupby('session_id')['hover_duration'].sum()
    tmp.name = 'object_hover_sum'
    dfs.append(tmp)
    
    tmp = df.loc[df['event_name']  == 'object_hover', :].groupby('session_id')['hover_duration'].mean()
    tmp.name = 'object_hover_mean'
    dfs.append(tmp)
    
    tmp = df.loc[df['event_name']  == 'object_hover', :].groupby('session_id')['hover_duration'].std()
    tmp.name = 'object_hover_std'
    dfs.append(tmp)
    
    tmp = df.loc[df['event_name']  == 'object_hover', :].groupby('session_id')['hover_duration'].median()
    tmp.name = 'object_hover_median'
    dfs.append(tmp)
    
    tmp = df.loc[df['event_name']  == 'object_hover', :].groupby('session_id')['hover_duration'].max()
    tmp.name = 'object_hover_max'
    dfs.append(tmp)
    
    tmp = df.loc[df['event_name']  == 'map_hover', :].groupby('session_id')['hover_duration'].sum()
    tmp.name = 'map_hover_sum'
    dfs.append(tmp)
    
    tmp = df.loc[df['event_name']  == 'map_hover', :].groupby('session_id')['hover_duration'].mean()
    tmp.name = 'map_hover_mean'
    dfs.append(tmp)
    
    tmp = df.loc[df['event_name']  == 'map_hover', :].groupby('session_id')['hover_duration'].std()
    tmp.name = 'map_hover_std'
    dfs.append(tmp)
    
    tmp = df.loc[df['event_name']  == 'map_hover', :].groupby('session_id')['hover_duration'].median()
    tmp.name = 'map_hover_median'
    dfs.append(tmp)
    
    tmp = df.loc[df['event_name']  == 'map_hover', :].groupby('session_id')['hover_duration'].max()
    tmp.name = 'map_hover_max'
    dfs.append(tmp)
  

    # Number of times notebook open
    tmp = df.loc[df['event_name']  == 'notebook_click', :].groupby('session_id')['event_name'].count()
    tmp.name = 'total_notebook_click'
    dfs.append(tmp)

    # Time spent for each event_name
    EVENT_NAMES = ['navigate_click','person_click','cutscene_click','object_click', 'map_hover','notification_click','map_click','observation_click']
    for event_name in EVENT_NAMES:
        tmp = df.loc[df['event_name'] == event_name, :].groupby('session_id')['action_time'].count()
        tmp.name = event_name + '_count'
        dfs.append(tmp)
        
        tmp = df.loc[df['event_name'] == event_name, :].groupby('session_id')['action_time'].sum()
        tmp.name = event_name + '_time_sum'
        dfs.append(tmp)
        
        tmp = df.loc[df['event_name'] == event_name, :].groupby('session_id')['action_time'].mean()
        tmp.name = event_name + '_time_mean'
        dfs.append(tmp)
        
        tmp = df.loc[df['event_name'] == event_name, :].groupby('session_id')['action_time'].std()
        tmp.name = event_name + '_time_std'
        dfs.append(tmp)
        
        tmp = df.loc[df['event_name'] == event_name, :].groupby('session_id')['action_time'].median()
        tmp.name = event_name + '_time_median'
        dfs.append(tmp)
        
        tmp = df.loc[df['event_name'] == event_name, :].groupby('session_id')['action_time'].max()
        tmp.name = event_name + '_time_max'
        dfs.append(tmp)

    # Count and time spent for each name
    NAMES = ['basic', 'close','next', 'open', 'prev', 'undefined']
    for name in NAMES:
        tmp = df.loc[df['name'] == name, :].groupby('session_id')['action_time'].count()
        tmp.name = name + '_time_count'
        dfs.append(tmp)
        
        tmp = df.loc[df['name'] == name, :].groupby('session_id')['action_time'].sum()
        tmp.name = name + '_time_sum'
        dfs.append(tmp)
        
        tmp = df.loc[df['name'] == name, :].groupby('session_id')['action_time'].mean()
        tmp.name = name + '_time_mean'
        dfs.append(tmp)
        
        tmp = df.loc[df['name'] == name, :].groupby('session_id')['action_time'].std()
        tmp.name = name + '_time_std'
        dfs.append(tmp)
        
        tmp = df.loc[df['name'] == name, :].groupby('session_id')['action_time'].median()
        tmp.name = name + '_time_median'
        dfs.append(tmp)
        
        tmp = df.loc[df['name'] == name, :].groupby('session_id')['action_time'].max()
        tmp.name = name + '_time_max'
        dfs.append(tmp)
        
        
    # Count and time per room_fqid
    if stage == 1:
        room_fqid_list = ['tunic.historicalsociety.closet', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.collection', 'tunic.kohlcenter.halloffame', 'tunic.capitol_0.hall']
    elif stage == 2:
        room_fqid_list = ['tunic.historicalsociety.basement', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.stacks', 'tunic.capitol_0.hall']
    elif stage == 3:
        room_fqid_list = ['tunic.historicalsociety.basement', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.stacks']
    
    for room_fqid in room_fqid_list:
        tmp = df.loc[df['room_fqid'] == room_fqid, :].groupby('session_id')['action_time'].count()
        tmp.name = room_fqid + 'time_count'
        dfs.append(tmp)
        
        tmp = df.loc[df['room_fqid'] == room_fqid, :].groupby('session_id')['action_time'].sum()
        tmp.name = room_fqid + 'time_sum'
        dfs.append(tmp)

        tmp = df.loc[df['room_fqid'] == room_fqid, :].groupby('session_id')['action_time'].mean()
        tmp.name = room_fqid + 'time_mean'
        dfs.append(tmp)

        tmp = df.loc[df['room_fqid'] == room_fqid, :].groupby('session_id')['action_time'].std()
        tmp.name = room_fqid + 'time_std'
        dfs.append(tmp)
        
        tmp = df.loc[df['room_fqid'] == room_fqid, :].groupby('session_id')['action_time'].median()
        tmp.name = room_fqid + 'time_median'
        dfs.append(tmp)
        
        tmp = df.loc[df['room_fqid'] == room_fqid, :].groupby('session_id')['action_time'].max()
        tmp.name = room_fqid + 'time_max'
        dfs.append(tmp)
        
        
    # Count and time per fqid
    if stage == 1:
        fqid_list = ['cs', 'gramps', 'groupconvo', 'notebook', 'plaque', 'plaque.face.date', 'retirement_letter', 'teddy', 'toentry', 'togrampa', 'tomap', 'tunic']
    elif stage == 2:    
        fqid_list = ['archivist', 'businesscards', 'businesscards.card_0.next', 'businesscards.card_1.next', 'businesscards.card_bingo.bingo', 'chap2_finale_c', 'gramps', 'journals',
                     'journals.hub.topics', 'journals.pic_0.next', 'journals.pic_1.next', 'journals.pic_2.bingo', 'logbook', 'logbook.page.bingo', 'magnify', 'reader', 'reader.paper0.next',
                     'reader.paper1.next', 'reader.paper2.bingo', 'tobasement', 'toentry', 'tofrontdesk', 'tomap', 'tostacks', 'trigger_coffee', 'trigger_scarf', 'tunic.capitol_1', 'tunic.drycleaner',
                     'tunic.historicalsociety', 'tunic.humanecology', 'tunic.library', 'wellsbadge']
    elif stage == 3:
        fqid_list = ['archivist_glasses', 'boss', 'ch3start', 'chap4_finale_c', 'coffee', 'colorbook', 'confrontation', 'crane_ranger', 'directory', 'directory.closeup.archivist', 'expert', 'flag_girl',
                     'glasses', 'gramps', 'groupconvo_flag', 'journals_flag', 'journals_flag.hub.topics', 'journals_flag.pic_0.bingo', 'journals_flag.pic_0.next', 'key', 'lockeddoor', 'reader_flag',
                     'reader_flag.paper0.next', 'reader_flag.paper2.bingo', 'remove_cup', 'savedteddy', 'seescratches', 'teddy', 'tobasement', 'tocage', 'toentry', 'tofrontdesk', 'tomap', 'tostacks',
                     'tracks', 'tracks.hub.deer', 'tunic.capitol_2', 'tunic.drycleaner', 'tunic.flaghouse', 'tunic.historicalsociety', 'tunic.library', 'tunic.wildlife', 'unlockdoor']

    for fqid in fqid_list:
        tmp = df.loc[df['fqid'] == fqid, :].groupby('session_id')['action_time'].count()
        tmp.name = fqid + 'time_count'
        dfs.append(tmp)
        
        tmp = df.loc[df['fqid'] == fqid, :].groupby('session_id')['action_time'].sum()
        tmp.name = fqid + 'time_sum'
        dfs.append(tmp)
        
        tmp = df.loc[df['fqid'] == fqid, :].groupby('session_id')['action_time'].mean()
        tmp.name = fqid + 'time_mean'
        dfs.append(tmp)
        
        tmp = df.loc[df['fqid'] == fqid, :].groupby('session_id')['action_time'].std()
        tmp.name = fqid + 'time_std'
        dfs.append(tmp)
        
        tmp = df.loc[df['fqid'] == fqid, :].groupby('session_id')['action_time'].median()
        tmp.name = fqid + 'time_median'
        dfs.append(tmp)
        
        tmp = df.loc[df['fqid'] == fqid, :].groupby('session_id')['action_time'].max()
        tmp.name = fqid + 'time_max'
        dfs.append(tmp)
    
    
    # Count and time per text_fqid
    if stage == 1:
        textfqid_list = ['tunic.historicalsociety.closet.gramps.intro_0_cs_0', 'tunic.historicalsociety.closet.retirement_letter.hub', 'tunic.historicalsociety.closet.teddy.intro_0_cs_0',
                         'tunic.historicalsociety.entry.groupconvo','tunic.kohlcenter.halloffame.plaque.face.date','tunic.kohlcenter.halloffame.togrampa']
    elif stage == 2 :
        textfqid_list = ['tunic.drycleaner.frontdesk.logbook.page.bingo', 'tunic.drycleaner.frontdesk.worker.done', 'tunic.drycleaner.frontdesk.worker.hub',
                         'tunic.historicalsociety.closet_dirty.gramps.news', 'tunic.historicalsociety.closet_dirty.trigger_coffee', 'tunic.historicalsociety.closet_dirty.trigger_scarf',
                         'tunic.historicalsociety.closet_dirty.what_happened', 'tunic.historicalsociety.frontdesk.archivist.have_glass', 'tunic.historicalsociety.frontdesk.archivist.hello',
                         'tunic.historicalsociety.frontdesk.archivist.newspaper', 'tunic.historicalsociety.stacks.journals.pic_2.bingo', 'tunic.humanecology.frontdesk.worker.intro',
                         'tunic.library.frontdesk.worker.hello']
    elif stage == 3:
        textfqid_list = ['tunic.flaghouse.entry.flag_girl.hello', 'tunic.flaghouse.entry.flag_girl.symbol', 'tunic.historicalsociety.basement.ch3start', 'tunic.historicalsociety.basement.savedteddy',
                         'tunic.historicalsociety.basement.seescratches', 'tunic.historicalsociety.cage.confrontation', 'tunic.historicalsociety.cage.glasses.afterteddy',
                         'tunic.historicalsociety.cage.teddy.trapped', 'tunic.historicalsociety.cage.unlockdoor', 'tunic.historicalsociety.collection_flag.gramps.flag',
                         'tunic.historicalsociety.entry.boss.flag', 'tunic.historicalsociety.entry.directory.closeup.archivist', 'tunic.historicalsociety.entry.groupconvo_flag',
                         'tunic.historicalsociety.entry.wells.flag', 'tunic.historicalsociety.frontdesk.archivist_glasses.confrontation', 'tunic.historicalsociety.stacks.journals_flag.pic_0.bingo',
                         'tunic.library.frontdesk.worker.flag', 'tunic.library.frontdesk.worker.nelson']
        
    for textfqid in textfqid_list:
        tmp = df.loc[df['text_fqid'] == textfqid, :].groupby('session_id')['action_time'].count()
        tmp.name = textfqid + 'time_count'
        dfs.append(tmp)
        
        tmp = df.loc[df['text_fqid'] == textfqid, :].groupby('session_id')['action_time'].sum()
        tmp.name = textfqid + 'time_sum'
        dfs.append(tmp)
        
        tmp = df.loc[df['text_fqid'] == textfqid, :].groupby('session_id')['action_time'].mean()
        tmp.name = textfqid + 'time_mean'
        dfs.append(tmp)
        
        tmp = df.loc[df['text_fqid'] == textfqid, :].groupby('session_id')['action_time'].std()
        tmp.name = textfqid + 'time_std'
        dfs.append(tmp)
        
        tmp = df.loc[df['text_fqid'] == textfqid, :].groupby('session_id')['action_time'].median()
        tmp.name = textfqid + 'time_median'
        dfs.append(tmp)
        
        tmp = df.loc[df['text_fqid'] == textfqid, :].groupby('session_id')['action_time'].max()
        tmp.name = textfqid + 'time_max'
        dfs.append(tmp)
        
    _train = pd.concat(dfs,axis=1).reset_index()
    
    
    # Time per level - Sum, Mean, Median, Std, Max, Count
    tmp = df.groupby(['session_id', 'level']).agg({'action_time' : ['sum', 'mean', 'median', 'std', 'max', 'count']}).reset_index()
    tmp.columns = tmp.columns.map(''.join)
    tmp_pivot = tmp.pivot_table(index='session_id', columns='level', values=['action_timesum', 'action_timemean', 'action_timemedian', 'action_timestd', 'action_timemax', 'action_timecount'])
    tmp_pivot.columns = [i[0] + '_' + str(i[1]) for i in tmp_pivot.columns]
    tmp_pivot = tmp_pivot.reset_index()
    
    if train == False:
        if stage == 1:
            level_range = range(0,5)
        elif stage == 2:
            level_range = range(5,13)
        elif stage == 3:
            level_range = range(13,23)
            
        NEEDED_COLS = ['session_id'] + ['action_timemax_' + str(i) for i in level_range] + ['action_timemean_' + str(i) for i in level_range] + ['action_timemedian_' + str(i) for i in level_range] +\
        ['action_timestd_' + str(i) for i in level_range] + ['action_timesum_' + str(i) for i in level_range] + ['action_timecount_' + str(i) for i in level_range]
        missing_cols = np.array(NEEDED_COLS[1:])[~np.isin(NEEDED_COLS[1:], tmp_pivot.columns)]
        for _col in missing_cols:
            tmp_pivot[_col] = 0
        tmp_pivot = tmp_pivot[NEEDED_COLS]
        
    _train = pd.merge(left=_train, right=tmp_pivot, on='session_id', how='left')
    
    
    # Time per level and event
    COLS = []
    EVENT_AT_LEVEL_STG1 = {
        'level0_events' : ['navigate_click', 'object_click', 'person_click'],
        'level1_events' : ['cutscene_click', 'navigate_click', 'object_click'],
        'level2_events' : ['cutscene_click', 'navigate_click', 'notification_click', 'object_click'],
        'level3_events' : ['cutscene_click', 'navigate_click', 'notification_click', 'object_click'],
        'level4_events' : ['navigate_click']
    }

    EVENT_AT_LEVEL_STG2 = {
        'level5_events' : ['cutscene_click', 'navigate_click'],
        'level6_events' : ['cutscene_click', 'navigate_click', 'person_click'],
        'level7_events' : ['navigate_click', 'object_click', 'person_click'],
        'level8_events' : ['navigate_click', 'notification_click', 'object_click', 'person_click'],
        'level9_events' : ['navigate_click', 'notification_click', 'object_click', 'person_click'],
        'level10_events' : ['navigate_click', 'person_click'],
        'level11_events' : ['navigate_click', 'notification_click', 'object_click'],
        'level12_events' : ['navigate_click']
    }

    EVENT_AT_LEVEL_STG3 = {
        'level13_events' : ['cutscene_click', 'navigate_click'],
        'level14_events' : ['navigate_click'],
        'level15_events' : ['person_click'],
        'level16_events' : ['cutscene_click', 'navigate_click'],
        'level17_events' : ['cutscene_click', 'navigate_click'],
        'level18_events' : ['person_click', 'navigate_click', 'object_click'],
        'level19_events' : ['navigate_click', 'object_click'],
        'level20_events' : ['navigate_click', 'object_click'],
        'level21_events' : ['navigate_click', 'notification_click', 'object_click'],
        'level22_events' : ['navigate_click']
    }


    tmp = df.groupby(['session_id', 'level', 'event_name']).agg({'action_time' : ['sum', 'mean', 'std', 'median', 'max', 'count']}).reset_index()
    tmp.columns = tmp.columns.map(''.join)
    tmp_pivot = tmp.pivot_table(index='session_id', columns=['level', 'event_name'], values=['action_timesum', 'action_timemean', 'action_timestd', 'action_timemedian', 'action_timemax',
                                                                                             'action_timecount'])
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
        COLS.extend([str(level) + '_' + i + '_' + 'action_timestd' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemedian' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemax' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timecount' for i in cols])


    if train == False:
        missing_cols = np.array(COLS)[~np.isin(COLS, tmp_pivot.columns)]
        for _col in missing_cols:
            tmp_pivot[_col] = 0


    tmp_pivot = tmp_pivot[COLS]
    tmp_pivot = tmp_pivot.reset_index()
    _train = pd.merge(left=_train, right=tmp_pivot, on='session_id', how='left')
    
    
    # Time per level and room
    COLS = []
    ROOM_AT_LEVEL_STG1 = {
            'level0_rooms' : ['tunic.historicalsociety.closet'],
            'level1_rooms' : ['tunic.historicalsociety.basement', 'tunic.historicalsociety.closet', 'tunic.historicalsociety.entry'],
            'level2_rooms' : ['tunic.historicalsociety.collection', 'tunic.historicalsociety.entry'],
            'level3_rooms' : ['tunic.historicalsociety.collection', 'tunic.historicalsociety.entry', 'tunic.kohlcenter.halloffame'],
            'level4_rooms' : ['tunic.capitol_0.hall', 'tunic.kohlcenter.halloffame']
        }

    ROOM_AT_LEVEL_STG2 = {
            'level5_rooms' : ['tunic.capitol_0.hall','tunic.historicalsociety.closet_dirty'],
            'level6_rooms' : ['tunic.historicalsociety.closet_dirty', 'tunic.historicalsociety.frontdesk'],
            'level7_rooms' : ['tunic.historicalsociety.entry', 'tunic.historicalsociety.frontdesk', 'tunic.humanecology.frontdesk'],
            'level8_rooms' : ['tunic.drycleaner.frontdesk', 'tunic.humanecology.frontdesk'],
            'level9_rooms' : ['tunic.drycleaner.frontdesk', 'tunic.library.frontdesk', 'tunic.library.microfiche'],
            'level10_rooms' : ['tunic.library.frontdesk', 'tunic.library.microfiche'],
            'level11_rooms' : ['tunic.historicalsociety.frontdesk', 'tunic.historicalsociety.stacks', 'tunic.library.frontdesk'],
            'level12_rooms' : ['tunic.capitol_1.hall', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.stacks']
        }

    ROOM_AT_LEVEL_STG3 = {
            'level13_rooms' : ['tunic.capitol_1.hall', 'tunic.historicalsociety.basement'],
            'level14_rooms' : ['tunic.historicalsociety.cage'],
            'level15_rooms' : ['tunic.historicalsociety.cage', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.frontdesk'],
            'level16_rooms' : ['tunic.historicalsociety.cage', 'tunic.historicalsociety.frontdesk'],
            'level17_rooms' : ['tunic.historicalsociety.basement', 'tunic.historicalsociety.cage', 'tunic.historicalsociety.collection_flag', 'tunic.historicalsociety.entry'],
            'level18_rooms' : ['tunic.historicalsociety.collection_flag', 'tunic.historicalsociety.entry', 'tunic.wildlife.center'],
            'level19_rooms' : ['tunic.flaghouse.entry', 'tunic.wildlife.center'],
            'level20_rooms' : ['tunic.flaghouse.entry', 'tunic.library.frontdesk', 'tunic.library.microfiche'],
            'level21_rooms' : ['tunic.historicalsociety.frontdesk', 'tunic.historicalsociety.stacks', 'tunic.library.frontdesk', 'tunic.library.microfiche'],
            'level22_rooms' : ['tunic.capitol_2.hall', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.stacks']
        }



    tmp = df.groupby(['session_id', 'level', 'room_fqid']).agg({'action_time' : ['sum', 'mean', 'std', 'median', 'max', 'count']})
    tmp.columns = tmp.columns.map(''.join)
    tmp_pivot = tmp.pivot_table(index='session_id', columns=['level', 'room_fqid'], values=['action_timesum', 'action_timemean', 'action_timestd', 'action_timemedian', 'action_timemax', 
                                                                                            'action_timecount'])
    tmp_pivot.columns = [str(i[1]) + '_' + i[2] + '_' + i[0] for i in tmp_pivot.columns]

    if stage == 1:
        level_range = range(0,5)
        ROOM_AT_LEVEL = ROOM_AT_LEVEL_STG1
    elif stage == 2:
        level_range = range(5,13)
        ROOM_AT_LEVEL = ROOM_AT_LEVEL_STG2
    else:
        level_range = range(13,23)
        ROOM_AT_LEVEL = ROOM_AT_LEVEL_STG3

    for level, cols in zip(list(level_range), ROOM_AT_LEVEL.values()):
        COLS.extend([str(level) + '_' + i + '_' + 'action_timesum' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemean' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timestd' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemedian' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timemax' for i in cols])
        COLS.extend([str(level) + '_' + i + '_' + 'action_timecount' for i in cols])


    if train == False:
        missing_cols = np.array(COLS)[~np.isin(COLS, tmp_pivot.columns)]
        for _col in missing_cols:
            tmp_pivot[_col] = 0


    tmp_pivot = tmp_pivot[COLS]
    tmp_pivot = tmp_pivot.reset_index()
    _train = pd.merge(left=_train, right=tmp_pivot, on='session_id', how='left')
    
    
    # Time per level and name
    tmp = df.groupby(['session_id', 'level', 'name']).agg({'action_time' : ['sum', 'mean', 'std', 'median', 'max', 'count']})
    tmp.columns = tmp.columns.map(''.join)
    tmp_pivot = tmp.pivot_table(index='session_id', columns=['level', 'name'], values=['action_timesum', 'action_timemean', 'action_timestd', 'action_timemedian', 'action_timemax', 'action_timecount'])
    tmp_pivot.columns = [str(i[1]) + '_' + i[2] + '_' + i[0] for i in tmp_pivot.columns]

    if stage == 1:
        cols_needed = ['0_basic', '0_undefined', '1_basic', '1_undefined', '2_basic', '2_undefined', '3_basic', '3_undefined', '4_basic', '4_open', '4_prev', '4_undefined']
    elif stage == 2:
        cols_needed = ['5_basic', '5_undefined', '6_basic',  '6_undefined', '7_basic', '7_undefined', '8_basic',  '8_undefined', '9_basic',  '9_undefined', '10_basic', '10_undefined', '11_basic', 
                       '11_undefined', '12_basic', '12_open', '12_prev', '12_undefined']
    elif stage == 3:
        cols_needed = ['13_basic', '13_undefined', '14_basic', '14_undefined', '15_basic', '15_undefined', '16_basic', '16_undefined', '17_basic', '17_undefined', '18_basic', '18_close', '18_undefined', 
                       '19_basic', '19_close', '19_undefined', '20_basic', '20_undefined', '21_basic', '21_close', '21_undefined', '22_basic', '22_undefined']

    COLS = []
    COLS.extend([i + '_action_timesum' for i in cols_needed])
    COLS.extend([i + '_action_timemean' for i in cols_needed])
    COLS.extend([i + '_action_timestd' for i in cols_needed])
    COLS.extend([i + '_action_timemedian' for i in cols_needed])
    COLS.extend([i + '_action_timemax' for i in cols_needed])
    COLS.extend([i + '_action_timecount' for i in cols_needed])

    if train == False:
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
        df = df.loc[df['keep'], ['session_id', 'action_time', 'event_name', 'room_fqid']]

        action_time_name = f'event_{str(num_id)}_time_'
        event_count_name = f'event_{str(num_id)}_'

        if rm_count:
            room_count_name = f'room_{str(num_id)}_'
            out = df.rename(columns={'action_time' : action_time_name, 'event_name' : event_count_name, 'room_fqid' : room_count_name})
            out = out.groupby('session_id').agg({event_count_name : 'count', room_count_name: 'nunique', action_time_name: ['sum', 'mean', 'std']}).reset_index()
        else:
            out = df.rename(columns={'action_time' : action_time_name, 'event_name' : event_count_name})
            out = out.groupby('session_id').agg({event_count_name : 'count', action_time_name: ['sum', 'mean', 'std']}).reset_index()

        out.columns = out.columns.map(''.join)
    
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
        time_taken_sum = df['action_time'].sum()
        time_taken_mean = df['action_time'].mean()
        time_taken_std = df['action_time'].std()
        if np.isnan(time_taken_std):
            time_taken_std = 0
        
        event_count = df['event_name'].count()
        
        if rm_count:
            room_uniq_count = df['room_fqid'].nunique()
            out = pd.DataFrame({f'event_{str(num_id)}_count' : event_count,
                                f'room_{str(num_id)}_nunique' : room_uniq_count,
                                f'event_{str(num_id)}_time_sum' : time_taken_sum, 
                                f'event_{str(num_id)}_time_mean' : time_taken_mean,
                                f'event_{str(num_id)}_time_std' : time_taken_std}, 
                                index=[0])
                             
        else:
            out = pd.DataFrame({f'event_{str(num_id)}_count' : event_count,
                                f'event_{str(num_id)}_time_sum' : time_taken_sum, 
                                f'event_{str(num_id)}_time_mean' : time_taken_mean,
                                f'event_{str(num_id)}_time_std' : time_taken_std}, 
                                index=[0])
            
    return out
    

def arrow_click(df, substring, name, train=True):
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
    
    _name = name + '_' + 'time'
    tmp2 = df.loc[cri_time, :].rename(columns={'action_time' : _name}).groupby('session_id').agg({_name : ['sum', 'mean', 'std']}).reset_index()
    tmp2.columns = tmp2.columns.map(''.join)
    out = pd.merge(tmp, tmp2, left_on='session_id', right_on='session_id', how='left')

    if train == False and len(out.columns) != 5:
        std_dev_col = _name + 'std'
        out[std_dev_col] = 0 
    return out


def point_click(df, fqid, fqid_bingo, name, train=True):
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
    tmp = df.groupby('session_id')[[name]].sum().reset_index()

    _name = name + '_' + 'time'
    tmp2 = df.loc[cri_time, :].rename(columns={'action_time' : _name}).groupby('session_id').agg({_name : ['sum', 'mean', 'std']}).reset_index()
    tmp2.columns = tmp2.columns.map(''.join)
    out = pd.merge(tmp, tmp2, left_on='session_id', right_on='session_id', how='left')

    if train == False and len(out.columns) != 5:
        std_dev_col = _name + 'std'
        out[std_dev_col] = 0 
    
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

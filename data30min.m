function data30 = load_PAC_30min(fileName)

    %% --- Lecture CSV propre -----------------------------------------------
    fid = fopen(fileName,'r');
    raw = textscan(fid,'%s %f','Delimiter',',','HeaderLines',1);
    fclose(fid);

    rawTime = string(raw{1});
    PAC     = raw{2};

    % Supprimer le fuseau horaire +00:00
    rawTime = extractBefore(rawTime, '+');

    % Conversion en datetime
    datetime_vec = datetime(rawTime,'InputFormat','yyyy-MM-dd HH:mm:ss');

    % Vérification
    assert(length(datetime_vec) == length(PAC), ...
        'datetime et PAC doivent avoir le même nombre de lignes');

    % Création table
    dataTable = table(datetime_vec, PAC, ...
        'VariableNames', {'Datetime','PAC'});

    %% --- Rééchantillonnage 30 min ----------------------------------------
    dataTable = sortrows(dataTable,'Datetime');

    TT = table2timetable(dataTable,'RowTimes','Datetime');

    TT_30min = retime(TT,'regular','mean','TimeStep',minutes(30));

    data30 = timetable2table(TT_30min,'ConvertRowTimes',true);

end
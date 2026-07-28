function D = prepare_supervised(data, dt_all, P, FH)
% Construit le probleme supervise pour un horizon donne et le prepare pour
% toutes les analyses. Tout est calcule de facon causale : la normalisation
% et les statistiques ne voient que la calibration.
%
%   data   : serie de puissance PV 30 min (colonne)
%   dt_all : timestamps alignes avec data (pour les features temporelles)
%   P      : parametres (voir main.m)
%   FH     : horizon en pas de 30 min
%
% Renvoie une structure D avec les jeux train/test, les features temporelles,
% les denominateurs de persistance et ce qu'il faut aux references.
    LB     = P.LB;
    n_temp = 4 * P.USE_TEMPORAL;

    % Representation a retards decales : X = 48 puissances passees, y = cible.
    [PVin, PVout] = sertomat(data, LB, FH);

    % Split chronologique AVANT normalisation (aucune fuite via mu/sd).
    idx_split = floor(P.ratio * size(PVin, 1));

    mu = mean(PVin(1:idx_split, :));      % stats sur la calibration seule
    sd = std(PVin(1:idx_split, :));
    sd(sd == 0) = 1;                       % colonnes constantes -> pas de division par 0
    PVn = (PVin - mu) ./ sd;

    D.X_train = PVn(1:idx_split, :);
    D.X_test  = PVn(idx_split+1:end, :);
    D.y_train = PVout(1:idx_split);
    D.y_test  = PVout(idx_split+1:end);
    D.n_test  = numel(D.y_test);

    % Features temporelles, calculees sur le timestamp de la CIBLE y(t+h).
    % Elles augmentent les entrees des modeles ELM/AR mais ne passent jamais
    % par la reduction de dimension (qui ne compresse que les 48 retards).
    if P.USE_TEMPORAL
        tgt_idx = (1:size(PVin,1))' + LB + FH - 1;
        dt      = dt_all(tgt_idx);
        hod     = hour(dt) + minute(dt)/60;             % heure decimale [0,24)
        doy     = day(dt, 'dayofyear');                 % 1..366
        TEMP    = [sin(2*pi*hod/24),     cos(2*pi*hod/24), ...
                   sin(2*pi*doy/365.25), cos(2*pi*doy/365.25)];
    else
        TEMP = zeros(size(PVin,1), 0);
    end
    D.TEMP_train = TEMP(1:idx_split, :);
    D.TEMP_test  = TEMP(idx_split+1:end, :);

    % Denominateurs NICE^k : erreurs de la persistance simple sur le test.
    % PVin(:,end) = y(t) donc la persistance simple aura NICE^k = 1 (definition).
    D.Persis_simple_test = PVin(idx_split+1:end, end);
    D.MAE_P  = compute_Lk_error(D.y_test, D.Persis_simple_test, 1);
    D.RMSE_P = compute_Lk_error(D.y_test, D.Persis_simple_test, 2);
    D.RMCE_P = compute_Lk_error(D.y_test, D.Persis_simple_test, 3);
    D.mean_y_test = mean(D.y_test);

    % Elements necessaires aux persistances cyclique / blend.
    D.data        = data;
    D.offset_base = idx_split + LB + FH - 1;
    D.idx_split   = idx_split;

    % Metadonnees pratiques.
    D.FH          = FH;
    D.LB          = LB;
    D.n_temp      = n_temp;
    D.LB_days     = LB / 48;
    D.FH_hours    = FH * 0.5;
    D.nParams_full = P.N_ELM_hidden * (LB + n_temp) + 2 * P.N_ELM_hidden;
end

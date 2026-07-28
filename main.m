clc; clear; close all;

% Tout est relatif a l'emplacement de main.m : l'utilisateur n'a RIEN a
% configurer (ni le chemin de la drtoolbox, ni l'emplacement des donnees).
% Il place le dossier ou il veut et lance main, c'est tout.
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end     % repli si execute cellule par cellule
cd(here);

%% ========================================================================
%  Reduction de dimension du signal PV endogene : diagnostics filter/wrapper
%  ------------------------------------------------------------------------
%  Point d'entree unique. On decrit ici TOUS les parametres, on charge les
%  donnees une fois, puis pour chaque horizon on calcule :
%     - les references (persistances, blend, ELM plein, AR plein)   [partagees]
%     - l'embedding de chaque methode de reduction                  [partage]
%     - l'analyse wrapper  (ELM + AR sur l'espace reduit)  -> Results_wrapper
%     - l'analyse filter   (score geometrique + seuil)     -> Results_filter
%  Les references sont calculees une seule fois par horizon et inserees a
%  l'identique dans les deux tables : filter et wrapper partagent les memes
%  baselines et le meme espace reduit.
%
%  Refs : methodes    https://lvdmaaten.github.io/drtoolbox/
%         blend/cyclo https://doi.org/10.1016/j.apm.2026.116988
%         NICEk       https://doi.org/10.1016/j.seta.2025.104588
%% ========================================================================

%% ---- Parametres --------------------------------------------------------
P.SMOKE_TEST = true;          % true = test rapide ; false = run complet

if P.SMOKE_TEST
    P.Ndata            = 1000;
    P.FH_list          = [1, 12];
    P.N_ELM_candidates = 2;
    P.N_ELM_hidden     = 8;
    P.L_MAX            = 400;
    P.filter_nmax      = 400;
else
    P.Ndata            = round(2 * 365.25 * 48);   % ~2 ans
    P.FH_list          = [1, 2, 6, 12, 20];        % 0.5 h a 10 h
    P.N_ELM_candidates = 500;
    P.N_ELM_hidden     = 100;
    P.L_MAX            = 3000;    % landmarks pour l'ajustement des methodes non lineaires
    P.filter_nmax      = 3000;    % sous-echantillon pour le score geometrique (O(n^2))
end

P.LB          = 48;            % fenetre d'entree (1 jour a 30 min)
P.ratio       = 0.50;          % part de calibration (split chronologique)
P.T_period    = 48;            % periode diurne en pas de 30 min
P.techniques  = {'PCA', 'KernelPCA', 'Isomap', ...
                 'LLE', 'Laplacian', 'DiffusionMaps', 'Autoencoder'};
P.drtoolbox_path = fullfile(here, 'drtoolbox');   % chemin absolu (visible par les workers parfor)
P.dataDir     = 'data';                            % sous-dossier des donnees (repli : racine du projet)
P.fileName    = 'PV_AC_20200801_20250706_Palaiseau.csv';   % un site ; format a conserver

P.KNN_OOS     = 12;            % voisins pour l'extension kNN (DiffusionMaps / repli)
P.SEED        = 42;            % reproductibilite (landmarks, sous-echantillons)
P.USE_TEMPORAL = true;         % ajoute heure + jour (sin/cos) aux entrees ELM/AR
P.ridge       = 0;             % regularisation Tikhonov de l'ELM (0 = pinv)

P.filter_alpha = 0.6;          % poids local (trustworthiness) dans S(d)
P.filter_k     = 10;           % voisins pour la trustworthiness
P.filter_q     = 0.95;         % seuil : plus petite d telle que S(d) >= q*max S

P.run_wrapper = true;
P.run_filter  = true;

addpath(genpath(P.drtoolbox_path));   % drtoolbox (van der Maaten) — sous-dossiers inclus
addpath(here);                        % fonctions utilitaires (aussi pour les workers parfor)
rng(P.SEED);

if P.SMOKE_TEST, fprintf('*** MODE SMOKE ***\n'); else, fprintf('*** MODE COMPLET ***\n'); end

%% ---- Donnees -----------------------------------------------------------
% Resolution automatique du fichier de donnees : d'abord dans data/, sinon a
% la racine du projet. L'utilisateur n'a aucun chemin a renseigner.
dataPath = fullfile(here, P.dataDir, P.fileName);
if ~isfile(dataPath), dataPath = fullfile(here, P.fileName); end
if ~isfile(dataPath)
    error(['Fichier de donnees introuvable : %s\n' ...
           'Placez-le dans %s%s%s ou a la racine du projet.'], ...
           P.fileName, P.dataDir, filesep, P.fileName);
end
data30 = data30min(dataPath);
data   = data30{1:P.Ndata, 2};       % puissance PV post-onduleur
dt_all = data30{1:P.Ndata, 1};       % timestamps alignes (features temporelles)
fprintf('Donnees : %d points (%.2f ans)\n', numel(data), numel(data)/48/365.25);

%% ---- Boucle sur les horizons ------------------------------------------
Results_wrapper = table();
Results_filter  = table();
filter_curves   = struct('FH_hours', {}, 'curves', {});   % courbes S(d) (figure Pareto eventuelle)

for FH = P.FH_list
    fprintf('\n=== FH = %d (%.1f h) ===\n', FH, FH*0.5);

    D   = prepare_supervised(data, dt_all, P, FH);
    REF = compute_references(D, P);

    % Embedding de chaque methode, calcule UNE fois et partage filter/wrapper.
    fprintf('  embeddings...\n');
    nTech = numel(P.techniques);
    EMB   = dr_embed(P.techniques{1}, D.X_train, D.X_test, P);
    for i = 2:nTech
        EMB(i) = dr_embed(P.techniques{i}, D.X_train, D.X_test, P);
    end

    if P.run_wrapper
        W = DR_wrapper(D, EMB, P);
        Results_wrapper = [Results_wrapper; REF; W];
    end
    if P.run_filter
        [Fr, cv] = DR_filter(D, EMB, P);
        Results_filter = [Results_filter; REF; Fr];
        filter_curves(end+1) = struct('FH_hours', D.FH_hours, 'curves', cv); %#ok<SAGROW>
    end
end

%% ---- Mise en forme, affichage, sauvegarde -----------------------------
% Tri par horizon puis par nombre de parametres : on lit directement l'erreur
% en fonction de l'empreinte du modele (impact de la reduction de dimension).
if P.run_wrapper
    Results_wrapper = sortrows(Results_wrapper, {'FH_hours', 'N_params'});
    disp(' '); disp('===== WRAPPER (erreur vs nb de parametres) =====');
    disp(Results_wrapper);
    writetable(Results_wrapper, 'Results_wrapper.csv');
    save('Results_wrapper.mat', 'Results_wrapper');
end
if P.run_filter
    Results_filter = sortrows(Results_filter, {'FH_hours', 'N_params'});
    disp(' '); disp('===== FILTER (score geometrique + d*_F) =====');
    disp(Results_filter);
    writetable(Results_filter, 'Results_filter.csv');
    save('Results_filter.mat', 'Results_filter', 'filter_curves');
end

fprintf('\nTermine. Tables : Results_wrapper.csv / Results_filter.csv\n');

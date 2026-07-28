function [F, curves] = DR_filter(D, EMB, P)
% Analyse FILTER : evalue, sans aucun modele de prevision, dans quelle mesure
% chaque espace reduit preserve la geometrie de l'espace des retards. Pour
% chaque methode et chaque dimension d on calcule le score
%
%     S(d) = alpha * T(k) + (1 - alpha) * rho
%
% ou T(k) est la trustworthiness (preservation des voisinages locaux) et rho
% une correlation de type Mantel entre les distances par paires de l'espace
% original et de l'espace reduit (preservation de la structure globale).
%
% Choix de la dimension : S(d) croit mecaniquement vers d = LB (plus on garde
% de composantes, mieux on preserve la geometrie), donc un simple argmax
% renverrait la dimension pleine. On retient plutot la PLUS PETITE dimension
% qui atteint une fraction q du score maximal :
%
%     d*_F = min { d : S(d) >= q * max_d S(d) }.
%
% La courbe S(d) complete est renvoyee (2e sortie) pour tracer, au besoin, le
% compromis geometrie / frugalite.
%
% Calcule sur la calibration uniquement, et sur un sous-echantillon de
% P.filter_nmax points (le score est en O(n^2) ; on borne n comme dans le
% protocole d'origine).
    techniques = P.techniques;
    nTech = numel(techniques);
    tdims = 2:P.LB;   nDim = numel(tdims);
    n_temp = D.n_temp;   nh = P.N_ELM_hidden;
    alpha = P.filter_alpha;   kNN = P.filter_k;   q = P.filter_q;

    % Sous-echantillon de la calibration (reproductible, toutes saisons).
    nTr   = size(D.X_train, 1);
    rng(P.SEED + 1);
    idx_f = randperm(nTr, min(P.filter_nmax, nTr));
    Xf    = D.X_train(idx_f, :);
    N     = size(Xf, 1);

    % Distances dans l'espace ORIGINAL des retards (calculees une fois).
    DX   = squareform(pdist(Xf));
    [~, ordX] = sort(DX, 2);
    rankX = zeros(N, N);
    for i = 1:N, rankX(i, ordX(i, :)) = 1:N; end
    mask = triu(true(N), 1);
    vX   = DX(mask);

    F = table();
    curves = struct('method', {}, 'dims', {}, 'S', {}, 'tau', {});

    for i = 1:nTech
        E      = EMB(i);
        method = E.method;
        fprintf('    [filter]  %s\n', method);

        Srow = nan(1, nDim);
        parfor j = 1:nDim
            [Ytr_d, ~, ok] = emb_slice(E, j);
            if ~ok, continue; end
            Yf = Ytr_d(idx_f, :);
            DY = squareform(pdist(Yf));

            tr  = trust_from_rank(rankX, DY, kNN);   % voisinages locaux
            rho = corr(vX, DY(mask));                % structure globale (Mantel)
            Srow(j) = alpha * tr + (1 - alpha) * rho;
        end

        % tau (frugalite) associe a chaque dimension, pour la courbe / figure.
        tau = (tdims + n_temp + 2) / (P.LB + n_temp + 2);

        % Selection par seuil : plus petite dim atteignant q * max(S).
        Smax = max(Srow);
        jsel = find(Srow >= q * Smax, 1, 'first');
        if isempty(jsel), jsel = find(~isnan(Srow), 1, 'first'); end
        if isempty(jsel), jsel = 1; end
        dstar = tdims(jsel);
        Sstar = Srow(jsel);

        nP = nh * (dstar + n_temp) + 2 * nh;   % empreinte structurelle a d*_F
        F  = [F; result_row(method, D.LB_days, D.FH_hours, dstar, nP, ...
                  100*(1 - nP/D.nParams_full), Sstar, ...
                  NaN, NaN, NaN, NaN, NaN, NaN, NaN)];   % pas de prevision (geometrique)

        curves(end+1) = struct('method', method, 'dims', tdims, 'S', Srow, 'tau', tau); %#ok<AGROW>
    end
end

% -------------------------------------------------------------------------
function tr = trust_from_rank(rankX, DY, k)
% Trustworthiness (Venna & Kaski) : penalise les points devenus k-voisins dans
% l'espace reduit alors qu'ils ne l'etaient pas dans l'espace original, au
% prorata de leur rang original. rankX(i,p) = rang du point p dans l'ordre des
% distances vues depuis i (1 = lui-meme).
    N = size(DY, 1);
    [~, ordY] = sort(DY, 2);
    S = 0;
    for i = 1:N
        redk = ordY(i, 2:k+1);      % k plus proches voisins dans l'espace reduit
        rx   = rankX(i, redk);      % leur rang dans l'espace original
        S    = S + sum(rx(rx > k) - k);   % penalite des voisins "usurpateurs"
    end
    tr = 1 - 2 * S / (N * k * (2*N - 3*k - 1));
end

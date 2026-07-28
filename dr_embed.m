function E = dr_embed(method, X_train, X_test, P)
% Calcule l'embedding d'UNE methode de reduction de dimension. Le resultat est
% partage tel quel par les deux analyses (filter et wrapper) : elles evaluent
% donc rigoureusement le meme espace reduit.
%
% Deux regimes selon la methode :
%   - Emboitees (PCA, KernelPCA, Isomap, LLE, Laplacian, DiffusionMaps) :
%     un seul embedding calcule a la dimension maximale (LB), qu'on tranchera
%     ensuite colonne par colonne. Pour les methodes non lineaires, on ajuste
%     sur un sous-echantillon de landmarks tires dans X_train (borne la
%     decomposition propre a L x L) puis on etend a tous les points :
%         * KernelPCA / Isomap / LLE / Laplacian -> out_of_sample (natif),
%           avec repli sur oos_knn si la sortie n'est pas finie ;
%         * DiffusionMaps -> oos_knn (pas d'extension native).
%   - Autoencoder : embedding NON emboite, recalcule pour chaque dimension.
%
% Aucune fuite : l'ajustement ne voit que X_train ; X_test n'est que projete.
    maxdim   = P.LB;
    E.method = method;
    E.dims   = 2:P.LB;
    E.nested = ~strcmpi(method, 'Autoencoder');

    % Tous les champs sont initialises (memes champs quelle que soit la
    % methode) pour pouvoir ranger les embeddings dans un tableau de structures.
    E.Ytr = []; E.Yte = []; E.Ytr_cell = {}; E.Yte_cell = {}; E.Kdim = 0;

    % ---- Autoencoder : un entrainement par dimension ----
    if ~E.nested
        nD  = numel(E.dims);
        Ytr = cell(1, nD);
        Yte = cell(1, nD);
        drp = P.drtoolbox_path;
        dims = E.dims;
        parfor j = 1:nD
            addpath(genpath(drp));
            d = dims(j);
            try
                [ytr, map] = compute_mapping(X_train, 'Autoencoder', d);
                Ytr{j} = ytr;
                Yte{j} = out_of_sample(X_test, map);
            catch ME
                warning('[Autoencoder] dim=%d embedding KO : %s', d, ME.message);
            end
        end
        E.Ytr_cell = Ytr;
        E.Yte_cell = Yte;
        E.Kdim     = maxdim;
        return
    end

    % ---- Methodes emboitees ----
    E.Ytr = []; E.Yte = []; E.Kdim = 0;
    try
        if strcmpi(method, 'PCA')
            % Lineaire et peu couteux : ajuste sur tout X_train.
            [Ytr, map] = compute_mapping(X_train, 'PCA', maxdim);
            Yte = out_of_sample(X_test, map);
        else
            % Non lineaire : ajustement sur landmarks (tires sur toute la
            % periode de calibration, donc toutes saisons representees).
            nTr  = size(X_train, 1);
            rng(P.SEED);                              % landmarks reproductibles
            Lsel = randperm(nTr, min(P.L_MAX, nTr));
            Xl   = X_train(Lsel, :);

            if strcmpi(method, 'LLE')
                % La drtoolbox ne regularise le Gram local de LLE que si
                % k > no_dims ; on ajuste a no_dims = maxdim, donc on force
                % k > maxdim pour eviter un Gram singulier (poids NaN).
                [Yl, map] = compute_mapping(Xl, method, maxdim, maxdim + 1);
            else
                [Yl, map] = compute_mapping(Xl, method, maxdim);
            end

            if strcmpi(method, 'DiffusionMaps')
                Ytr = oos_knn(X_train, Xl, Yl, P.KNN_OOS);
                Yte = oos_knn(X_test,  Xl, Yl, P.KNN_OOS);
            else
                ok = false;
                try
                    Ytr = out_of_sample(X_train, map);
                    Yte = out_of_sample(X_test,  map);
                    ok  = all(isfinite(Ytr(:))) && all(isfinite(Yte(:)));
                catch ME2
                    warning('[%s] OOS natif exception : %s', method, ME2.message);
                end
                if ~ok
                    warning('[%s] OOS natif non fini -> repli kNN', method);
                    Ytr = oos_knn(X_train, Xl, Yl, P.KNN_OOS);
                    Yte = oos_knn(X_test,  Xl, Yl, P.KNN_OOS);
                end
            end
        end
        E.Ytr  = Ytr;
        E.Yte  = Yte;
        E.Kdim = size(Ytr, 2);
    catch ME
        warning('[%s] embedding KO : %s', method, ME.message);
    end
end

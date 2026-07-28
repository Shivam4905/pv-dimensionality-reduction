function [Ytr_d, Yte_d, ok] = emb_slice(E, j)
% Renvoie l'embedding a la dimension cible d = E.dims(j), pour l'apprentissage
% et le test. Pont commun entre dr_embed (qui calcule) et les analyses
% filter / wrapper (qui consomment), de sorte que les DEUX voient exactement
% le meme espace reduit.
%
%   - methodes emboitees (PCA, KernelPCA, Isomap, LLE, Laplacian, Diffusion) :
%     l'embedding est calcule une fois a dim = LB, on tranche les d premieres
%     colonnes.
%   - Autoencoder (non emboite) : un embedding distinct par dimension, stocke
%     dans des cellules.
    d = E.dims(j);
    if E.nested
        if isempty(E.Ytr) || E.Kdim < d
            Ytr_d = []; Yte_d = []; ok = false; return
        end
        Ytr_d = E.Ytr(:, 1:d);
        Yte_d = E.Yte(:, 1:d);
        ok = true;
    else
        Ytr_d = E.Ytr_cell{j};
        Yte_d = E.Yte_cell{j};
        ok = ~isempty(Ytr_d);
    end
end

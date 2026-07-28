function Y = oos_knn(points, Xland, Yland, k)
% Extension hors-echantillon par interpolation kNN ponderee (facon Nystrom).
% Pour chaque point on cherche ses k plus proches landmarks et on moyenne
% leurs coordonnees reduites, avec des poids en 1/distance.
%
% Utilisee pour Diffusion Maps (pas d'extension out-of-sample native dans la
% drtoolbox) et comme filet de securite quand l'OOS natif renvoie des NaN.
%
%   points : N x D  (a projeter)     Xland : L x D  (landmarks, espace source)
%   Yland  : L x d  (embedding)      k     : nb de voisins
    N  = size(points, 1);
    d  = size(Yland, 2);
    Y  = zeros(N, d);
    k  = min(k, size(Xland, 1));
    bb = sum(Xland.^2, 2)';                          % 1 x L
    for i = 1:N
        p     = points(i, :);
        dist2 = max(sum(p.^2) + bb - 2*(p * Xland'), 0);   % distances^2 aux landmarks
        [ds, idx] = mink(dist2, k);
        w = 1 ./ (sqrt(ds) + eps);
        Y(i, :) = (w / sum(w)) * Yland(idx, :);
    end
end

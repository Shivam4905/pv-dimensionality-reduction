function Lk = compute_Lk_error(y_true, y_pred, k)
% Erreur en norme L^k (moyenne des |erreurs|^k, puis racine k-ieme).
%   k=1 -> MAE, k=2 -> RMSE, k=3 -> insiste sur les grosses erreurs (rampes).
    Lk = mean(abs(y_true(:) - y_pred(:)) .^ k) .^ (1/k);
end

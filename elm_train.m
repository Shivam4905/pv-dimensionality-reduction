function [beta, IW, Bias] = elm_train(Z, y, nh, nc, ridge)
% Extreme Learning Machine : couche cachee a poids d'entree tires au hasard,
% poids de sortie resolus en forme fermee. On tire nc jeux de poids d'entree
% et on garde celui qui minimise le residu D'APPRENTISSAGE (pas de fuite du
% test dans ce choix).
%
%   Z     : entrees d'apprentissage (n x d, deja reduites + features temporelles)
%   y     : cible d'apprentissage (n x 1)
%   nh    : nombre de neurones caches
%   nc    : nombre de tirages candidats
%   ridge : regularisation Tikhonov (lambda) sur les poids de sortie ; 0 = pinv
%
% Sortie : beta (nh x 1), IW (nh x d), Bias (nh x 1). Prediction :
%   H = 1./(1+exp(-(Zt*IW' + Bias')));  y_hat = H*beta;
    d       = size(Z, 2);
    best    = inf;
    beta    = []; IW = []; Bias = [];
    for c = 1:nc
        IWc  = rand(nh, d) * 2 - 1;
        Bc   = rand(nh, 1);
        H    = 1 ./ (1 + exp(-(Z * IWc' + Bc')));
        if ridge > 0
            bc = (H' * H + ridge * eye(nh)) \ (H' * y);   % ELM regularise
        else
            bc = pinv(H) * y;                             % moindres carres min-norme
        end
        r = sqrt(mean((H * bc - y).^2));
        if r < best
            best = r; beta = bc; IW = IWc; Bias = Bc;
        end
    end
end

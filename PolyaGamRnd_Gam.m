function x = PolyaGamRnd_Gam(a,c,Truncation)
%Generating a Polya-Gamma random varaible using the summation of
%Truncation gamma random variables
%Making both the mean and variance unbiased
%Matlab code for "Mingyuan Zhou, Softplus regressions and convex polytopes,
%2016"
%http://mingyuanzhou.github.io/
%Copyright: Mingyuan Zhou, March 2016

a=full(a(:));
c=full(abs(c(:)));
idx = a~=0;
a = a(idx);
c = c(idx);

if ~isempty(a)
    idx_c = c~=0;
    c1 = c(idx_c);
    a1 = a(idx_c);
    
    xmeanfull = a/4;
    xmeanfull(idx_c) = a1.*(tanh(c1/2))./(2*c1);
    
    xvarfull=a/24;
    
    idx_c1 = c>=1e-3;
    c1 = c(idx_c1);
    a1 = a(idx_c1);
    
    xvarfull(idx_c1) = 0.5*exp(log(a1)-3*log(c1)+log(-expm1(-2*c1)-2*c1.*exp(-c1))-log(1+exp(-2*c1)+2*exp(-c1)));
    
    idx_c1 = c<1e-3;
    c1 = c(idx_c1);
    a1 = a(idx_c1);

    xvarfull(idx_c1) = 0.5*exp(log(a1)+ max(-3*log(c1)+log(-expm1(-2*c1)-2*c1.*exp(-c1))-log(1+exp(-2*c1)+2*exp(-c1)), ...
        -log(12)-2*logcosh(c1/2)));
    
    if Truncation>1
        temp = bsxfun(@plus,((1:Truncation-1)-0.5).^2,c.^2/4/pi^2);
        xmeantruncate = 1/2/pi^2*a.*sum(1./temp,2);
        xmean = full(max(xmeanfull - xmeantruncate,0));
        
        xvartruncate = 1/4/pi^4*a.*sum(1./(temp).^2,2);
        
        %x = 1/2/pi^2*sum(randg(a*ones(1,Truncation))./temp,2);
        x = 1/2/pi^2*sum(randg(a(:,ones(1,Truncation-1)))./temp,2);
%         x = 1/2/pi^2*sum(randg(repmat(a, 1, Truncation-1))./temp,2);

        xvar = full(max(xvarfull - xvartruncate,0));
        
        dex1 = xvarfull>=xvartruncate+realmin;
             
        x(~dex1) = x(~dex1) + xmean(~dex1);
        if nnz(dex1)>0
            %x(dex1)=x(dex1) + exp(log(max(randg(exp(2*log(xmean(dex1))-log(xvar(dex1)))),realmin))+log(max(xvar(dex1),realmin))-log(max(xmean(dex1),realmin)));
            x(dex1)=x(dex1) + randg((xmean(dex1)./max(sqrt(xvar(dex1)),realmin)).^2).*(xvar(dex1)./max(xmean(dex1),realmin));
            
           %exp(log(max()))),realmin))+log(max(,realmin))-log(max(,realmin)));
           %sig  = log1p(exp(log(xvar(dex1))-2*log(max(xmean(dex1),realmin))));
           
%            sig  = logOnePlusExp(log(xvar(dex1))-2*log(max(xmean(dex1),realmin)));
%            mu = log(xmean(dex1))-sig/2;
%            x(dex1) = x(dex1) + exp(randn(size(mu)).*sqrt(sig)+mu);
           
%              mu = xmean(dex1);
%              %lambda =   max((mu./max(xvar(dex1).^(1/3),realmin)).^3,realmin); % 
%              lambda = max(exp(3*log(max(mu,realmin))-log(xvar(dex1))),realmin); 
% %             
%              y0 = chi2rnd(1,size(mu,1),size(mu,2));
%              x0 = mu.*(1+(mu.*y0 - sqrt(4*mu.*lambda.*y0+(mu.*y0).^2))./(2*lambda));
%              out0 = x0;
%              dex = rand(size(x0))>mu./(mu+x0);
%              out0(dex)=exp(2*log(max(mu(dex),realmin))-log(max(x0(dex),realmin)));
%             % out0(dex)=(mu(dex)./sqrt(max(x0(dex),realmin))).^2;
%              x(dex1) = x(dex1)+out0;
        end
        
        
%         cc = max(xmean,realmin)./max(xvar,realmin);
%         aa = full(xmean.*cc);
%         x = x+ randg(aa)./max(cc,realmin);
        
%         dex =  xvarfull>=xvartruncate+realmin & xmean >= realmin;
%         if nnz(dex)>0
%             xvar = xvarfull(dex) - xvartruncate(dex);
%             cc = xmean(dex)./xvar;
%             aa = xmean(dex).*cc;
%             x(dex) = x(dex)+ randg(aa)./max(cc,realmin);
%             %x(dex) = x(dex)+ randg(xmean(dex).^2./xvar).*xvar./xmean(dex);
%         end
    else
        cc = xmeanfull./max(xvarfull,realmin);
        aa = xmeanfull.*cc;
        x = randg(aa)./max(cc,realmin);
        
%         sig  = logOnePlusExp(log(xvarfull)-2*log(max(xmeanfull,realmin)));
%         mu = log(xmeanfull)-sig/2;
%         x = exp(randn(size(mu)).*sqrt(sig)+mu);
        
    end
    temp=x;
    x = zeros(size(idx));
    x(idx)=temp;
else
    x = zeros(size(idx));
end

function y=logcosh(x)
y = abs(x)-log(2)+log1p(exp(-2*abs(x)));
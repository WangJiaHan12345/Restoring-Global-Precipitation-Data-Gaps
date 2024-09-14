SaveFolder=strcat('G:\��ظ�ԭ\γ��\','2015-2016'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end

fid = fopen('G:\��ظ�ԭ\�й�-��ظ�ԭ-440.txt','rb','l');
data = cell2mat(textscan(fid,'%f','headerlines',6));
data = reshape(data,700,440);
data = data'; 
fclose(fid); 
disp('������...');

% ����������txt�ļ�
for i = 1:1:440
    for j =1:1:700
        if data(i,j) >= 0
            
            a = 59 - (i-1)*0.1;
            rain = ones(731,1)*a;

    
            SaveFiles= [num2str(i,'%03d'),num2str(j,'%03d'),'.txt']; %CPC����ļ���·��

             outfile=strcat(SaveFolder,'\',SaveFiles);

             if exist(outfile,'file')~=0 
                delete(outfile);     
             end
     
     
             fid1=fopen(outfile,'w');
             
             for m=1:1:731  
                 for n=1:1:1
                     if n==1
                         fprintf(fid1,'%g\r\n',rain(m,n));
                     else
                         fprintf(fid1,'%g ',rain(m,n));
                     end
                 end   
             end
             
             fclose(fid1);             
        end  
    end
end
disp('�������');

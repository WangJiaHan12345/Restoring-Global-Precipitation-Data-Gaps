% ���������ʽ��Ϊ���ڵ���ʽ
FolderPath=input('���������ݴ洢�ļ���:','s');  %�����������
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
Files=dir(FolderPath);
FilesCount=length(Files);

SaveFolder=strcat('G:\ȫ��\ʱ��Ԥ����\�����ͼ��������\����\ֻ������վ��\','��ˮ'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end

fid = fopen('G:\ȫ��\ʱ��Ԥ����\�����ͼ��������\����\ԭʼ����\rain_sum_12-2.txt','rb','l');
data = cell2mat(textscan(fid,'%f','headerlines',0));
data = reshape(data,720,240);
data = data'; 
fclose(fid); 

result = [];
a = 0;
% ����������תΪ������
for k=3:FilesCount
       
      Name=Files(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
      
      
      i= str2num(Name(location(end)-6:location(end)-4));
      j= str2num(Name(location(end)-3:location(end)-1));
      
      a = a + 1;
      result(a,1) = data(i,j);
       
end


 outfile=strcat(SaveFolder,'\','san_sum_12-2.txt');

 if exist(outfile,'file')~=0 
    delete(outfile);     
 end


 fid1=fopen(outfile,'w');
     
 for i=1:1:a
     for j=1:1:1
         if j==1
             fprintf(fid1,'%g\r\n',result(i,j));
         else
             fprintf(fid1,'%g ',result(i,j));
         end
     end   
 end
 fclose(fid1);         
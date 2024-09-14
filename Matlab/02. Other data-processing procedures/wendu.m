SaveFolder=strcat('H:\ʱ��Ԥ��\�������\�¶�\','2018_1'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
disp('������...');

ncFilePath='H:\ʱ��Ԥ��\�������\�¶�\tmin.2018.nc';

lon=ncread(ncFilePath,'lon');%��ȡ���ȱ���
lat=ncread(ncFilePath,'lat');%��ȡγ�ȱ���
time=ncread(ncFilePath,'time');%��ȡʱ�����
tmin=ncread(ncFilePath,'tmin');%��ȡ��ɢ����������
for a=1:1:365
    tmin1=tmin(:,:,a);
    tmin1=rot90(tmin1,3);
    tmin1=fliplr(tmin1);

    for i=1:1:360
        for j=1:1:720
           if tmin1(i,j)== -9.969209968386869e+36
               tmin1(i,j)= -9999;
           end
        end
    end
      
      Date=datetime(2018,1,a);
      DateString = datestr(Date);

      SaveFiles=strcat(DateString,'.txt'); %CPC����ļ���·��
     
     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');

     fprintf(fid1,'NCOLS        720\r\nNROWS        240\r\nXLLCORNER   0\r\nYLLCORNER    -60\r\nCELLSIZE    0.5\r\nNODATA_VALUE    -9999\r\n');


     for i=61:1:300
         for j=1:1:720
             if j==720
                 fprintf(fid1,'%g\r\n',tmin1(i,j));
             else
                fprintf(fid1,'%g ',tmin1(i,j));
             end
         end   
     end
         fclose(fid1); 
end
disp('������');

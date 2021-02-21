load "/home/yqliu/miniconda3/pkgs/ncl-6.6.2-h297df27_19/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "/home/yqliu/miniconda3/pkgs/ncl-6.6.2-h297df27_19/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin

data=fbindirread("/home/yqliu/stas_vs/pred_bin_results/Pred-nonDate-28",0,(/64,4/),"float")
print(data(0,:))

lat=data(:,1)
lon=data(:,2)
rain=data(:,3)
lat1d=new((/161/),float)
lon1d=new((/161/),float)
lat2d=new((/161,161/),float)
lon2d=new((/161,161/),float)
rain@_FillValue=-9999.
;lat = fbindirread("lat.dat",0,(/52/),"float")
;lon = fbindirread("lon.dat",0,(/52/),"float")
;rain1 = fbindirread("rain1.dat",0,(/52/),"float")
;rain2 = fbindirread("rain2.dat",0,(/52/),"float")
;rain3 = fbindirread("rain3.dat",0,(/52/),"float")
;rain4 = fbindirread("rain4.dat",0,(/52/),"float")

;print(lat)
;print(lon)

do i = 0,160
  lon1d(i)=105.0+0.125*i
  lat1d(i)=20.0+0.125*i
  lon2d(:,i)=105.0+0.125*i
  lat2d(i,:)=20.0+0.125*i
end do

grid = new((/161,161/),float)

printVarSummary(rain)
;rscan = (/.5 , .3 , .15 , .05/) 
rscan = (/ 2, 1, 0.8, 0.5/)
rain = where(rain.gt.999997.5,rain@_FillValue,rain)
rain = where(rain.gt.999989.,0.05,rain)
grid = obj_anal_ic_deprecated(lon,lat,rain,lon1d,lat1d,rscan,False)

;fbindirwrite("grid.dat",grid)
;grid@_FillValue=-9999.9
;print(grid(:,100))
printVarSummary(grid)
grid@lat2d = lat2d
grid@lon2d = lon2d

wks = gsn_open_wks("png","pltrain_pred")
gsn_define_colormap(wks,"radar_1")
mpres = True
;mpres@mpGeophysicalLineColor      = "grey2"
mpres@mpFillOn = False
mpres@mpGeophysicalLineThicknessF = 1.0
mpres@mpDataBaseVersion="Ncarg4_1"
mpres@mpDataSetName="Earth..4"
mpres@mpOutlineOn = True
;mpres@mpUSStateLineColor          = "grey19"
mpres@mpUSStateLineThicknessF     = 2.0
mpres@mpGeophysicalLineThicknessF = 2.0

mpres@mpMaxLatF = 39                         ; specify the plot domain
mpres@mpMinLatF = 23                         ;                         
mpres@mpMinLonF = 113                        ;
mpres@mpMaxLonF = 122.5                        ;
;mpres@mpOutlineOn = True                     ; turn the map outline on
;mpres@mpFillOn = True
;mpres@mpMaskAreaSpecifiers = (/"China:Shanghai"/)
;mpres@mpOutlineSpecifiers =(/"shanghai shi"/)
mpres@mpOutlineSpecifiers=(/"China:states","Taiwan"/)
;mpres@mpAreaMaskingOn = True
;mpres@mpMaskAreaSpecifiers = (/"shanghai shi"/)
;mpres@mpFillDrawOrder = "PostDraw"
;mpres@mpOceanFillColor = "white"
mpres@gsnDraw   = False
mpres@gsnFrame   = False

;mpres@tmXBMode    =       "Explicit"
;mpres@tmXBValues  =  (/120.5,121,121.5,122/)
;mpres@tmXBLabels  =  (/"120.5","121","121.5","122"/)
;mpres@tmXBMinorValues = fspan(120.5,122.3,19)

;mpres@tmYLMode    =       "Explicit"
;mpres@tmYLValues  =  (/30.5,31,31.5,32/)
;mpres@tmYLLabels  =  (/"30.5","31","31.5","32"/)
;mpres@tmYLMinorValues = fspan(30.4,32.1,18)

;opts_z = mpres
;opts_z@cnLineColor = "Blue"
;opts_z@gsnContourLineThicknessesScale = 3.0
;opts_z@cnLevelSpacingF      =   20
;plot_h = gsn_contour(wks,h(12,:,:),opts_z)
;delete(opts_z)

;overlay(plot_h,vector)
;draw(plot_h)
;frame(wks)
opts = mpres
opts@cnFillOn = True
;opts@cnLevelSelectionMode = "ExplicitLevels"
;opts@cnLevels             = (/ 20.,30,40,50,60.,70., 80.,90.,100./)
;opts@cnFillColors         = (/101,97,94,90,84,79,70,65,59,52/)

opts@cnLevelSelectionMode = "ExplicitLevels"
opts@cnLevels             = (/0.1,4,13,25,60,120/)
opts@cnFillColors         = (/ 9, 13, 16, 11, 12, 23, 22/)
;opts@cnLevelSelectionMode = "ManualLevels"
;opts@cnMinLevelValF       = 0
;opts@cnMaxLevelValF       = 10
opts@cnLevelSpacingF =1
opts@cnLinesOn = False
opts@lbOrientation = "Vertical"
opts@pmLabelBarWidthF = 0.1
opts@pmLabelBarHeightF = 0.6
plot_rh = gsn_csm_contour_map(wks,grid,opts)

txres = True
txres@txFontHeightF = 0.012

;do i = 0, 51
;tx = floattoint(rain1(i))
;tex = "" + tx
;text = gsn_add_text(wks,plot_rh,tex(0),lon(i),lat(i),txres)
;delete(tex)
;end do

draw(plot_rh)
frame(wks)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

end

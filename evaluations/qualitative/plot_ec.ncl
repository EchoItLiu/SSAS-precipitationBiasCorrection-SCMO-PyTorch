load "/home/yqliu/miniconda3/pkgs/ncl-6.6.2-h297df27_19/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "/home/yqliu/miniconda3/pkgs/ncl-6.6.2-h297df27_19/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "/home/yqliu/miniconda3/pkgs/ncl-6.6.2-h297df27_19/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "/home/yqliu/miniconda3/pkgs/ncl-6.6.2-h297df27_19/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"
load "/home/yqliu/miniconda3/pkgs/ncl-6.6.2-h297df27_19/lib/ncarg/nclscripts/csm/contributed.ncl"

;starting
begin
;rain0=fbindirread("081112",0,(/125*77/),"float")  ;fbindirread(fname,0,(/nt*ny*nx/),"float"))
;rain=reshape(rain0(:),(/125,77/))

a_pre=addfile("/home/yqliu/vs_rain6/ec_file/C1D08020000080206001.grb","r")
a=addfile("/home/yqliu/vs_rain6/ec_file/C1D08020000080212001.grb","r")


print (a_pre)

names=getfilevarnames(a)
;names=getfilevarnames(a_pre)

print(names)


tp_pre = a_pre->TP_GDS0_SFC
tp = a->TP_GDS0_SFC
printVarSummary(tp_pre)
printVarSummary(tp)

tp_inveral_6 = tp - tp_pre
copy_VarAtts(tp, tp_inveral_6)
copy_VarCoords(tp,tp_inveral_6)

printVarSummary(tp_inveral_6)

  wks = gsn_open_wks("png","plt_ec")
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
opts@gsnLeftString = "6h precipitation STAS"
opts@gsnRightString = "20xx0xxxx"
plot_rh = gsn_csm_contour_map(wks, tp_inveral_6, opts)

draw(plot_rh)
frame(wks)
end


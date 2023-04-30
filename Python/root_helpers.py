import ROOT as R

def print_daughters(mdst, i_part, indent=0):
    """Given an index to an MC particle, print all the daughters.
    This function will recursively print the daughters of the daughters.
    Arguments:
        mdst   -- a MiniDst object that was linked to a TTree
        i_part -- the index of the particle to print
        ident  -- the amount of indentation of the output.
    """

    if mdst.mc_part_pdg.size() > 0:
        part_pdg = mdst.mc_part_pdg[i_part]
    else:
        part_pdg = 0
    print(" "*indent+f" {i_part:3d}  pdg: {part_pdg:4d}  E: {mdst.mc_part_energy[i_part]:9.6f} " +
          f"p = ({mdst.mc_part_px[i_part]:9.6f},{mdst.mc_part_py[i_part]:9.6f},{mdst.mc_part_pz[i_part]:9.6f})" +
          f"v = ({mdst.mc_part_x[i_part]:5.2f},{mdst.mc_part_y[i_part]:5.2f},{mdst.mc_part_z[i_part]:5.2f}) " +
          f"end=({mdst.mc_part_end_x[i_part]:5.2f},{mdst.mc_part_end_y[i_part]:5.2f},{mdst.mc_part_end_z[i_part]:5.2f})")
    if len(mdst.mc_part_daughters[i_part]) > 0:
        print(" "*(indent+14) + "| ")
        for i in range(len(mdst.mc_part_daughters[i_part])):
            ii = mdst.mc_part_daughters[i_part][i]  # Get the daughter reference
            print_daughters(mdst, ii, indent+11)            # Print by recursing


def print_mc_particle_tree(mdst):
    """Print the MCParticle tree.
    Arguments:
        mdst -- a MiniDst object that was linked to a TTree.
    """
    for i in range(len(mdst.mc_part_parents)):
        if len(mdst.mc_part_parents[i]) == 0:  # top level particle
            print_daughters(mdst, i, 0)


def get_vertex_dictionary():
    """Return a dictionary that translates the Vertex numbers to the
    ParticleType Enum name"""

    out = {
        0: "FINAL_STATE_PARTICLE_KF",
        1: "UC_V0_VERTICES_KF",
        2: "BSC_V0_VERTICES_KF",
        3: "TC_V0_VERTICES_KF",
        4: "UC_MOLLER_VERTICES_KF",
        5: "BSC_MOLLER_VERTICES_KF",
        6: "TC_MOLLER_VERTICES_KF",
        7: "OTHER_ELECTRONS_KF",
        8: "UC_VC_VERTICES_KF",
        0 + 9: "FINAL_STATE_PARTICLE_GBL",
        1 + 9: "UC_V0_VERTICES_GBL",
        2 + 9: "BSC_V0_VERTICES_GBL",
        3 + 9: "TC_V0_VERTICES_GBL",
        4 + 9: "UC_MOLLER_VERTICES_GBL",
        5 + 9: "BSC_MOLLER_VERTICES_GBL",
        6 + 9: "TC_MOLLER_VERTICES_GBL",
        7 + 9: "OTHER_ELECTRONS_GBL",
        8 + 9: "UC_VC_VERTICES_GBL",
    }
    return out


def fancy_plot(histo, ones_lb, opt=0):
    """
    Fancy plot of the Calorimeter with the 2D histo overlayed.
    With opt = 0x01 - Draw col instead of colz
    With opt = 0x02 -
    With opt = 0x04 - reset maximum.
    With opt = 0x06 - Keep stats box.
    """
    # this defines the position of the top right region of big boxes, others will fall in by symmetry
    ecal_x_first = 1
    ecal_nx_first = -1
    ecal_y_first = 1
    ecal_ny_first = -1

    ecal_nx = 23
    ecal_ny = 5

    xax = histo.GetXaxis()
    yax = histo.GetYaxis()

    # if R.gROOT.FindObject(histo.GetName()+"_oneslb"): # if this one exists all the rest probably exist too
    #     print(f'histogram {histo.GetName()+"_oneslb"} found.')
    #     ones_lb= R.gROOT.FindObject(histo.GetName()+"_oneslb")
    #     ones_lb.Clear()
    if ones_lb is None:
        print(f'Booking histogram {histo.GetName()+"_oneslb"}')
        ones_lb = R.TH2F(histo.GetName()+"_oneslb", "oneslb", (ecal_nx+1)*2+1, -ecal_nx-1.5,
                         ecal_nx+1.5, (ecal_ny+1)*2+1, -ecal_ny-1.5, ecal_ny+1.5);
    else:
        ones_lb.Clear()

    if opt & 0x4 == 0:
        histo.SetMaximum()
    if histo.GetMaximum() < 1:
        histo.SetMaximum(1.1)

    if opt & 0x6 == 0:
        histo.SetStats(0)

    SetMax = histo.GetMaximum()
    if SetMax < 1.1:
        SetMax = 1.1

    xax=ones_lb.GetXaxis()
    yax=ones_lb.GetYaxis()

    # this chunk of code just puts the grid in the right place
    for i in range(ecal_nx):
        for j in range(ecal_ny):
            ones_lb.SetBinContent(xax.FindBin(ecal_x_first+i), yax.FindBin(ecal_y_first+j), SetMax*10)
            ones_lb.SetBinContent(xax.FindBin(ecal_x_first+i), yax.FindBin(ecal_ny_first-j), SetMax*10)
            if j == 0 and 0 < i < 10:
                pass
            else:
                ones_lb.SetBinContent(xax.FindBin(ecal_nx_first-i), yax.FindBin(ecal_ny_first-j), SetMax*10)
                ones_lb.SetBinContent(xax.FindBin(ecal_nx_first-i), yax.FindBin(ecal_y_first+j), SetMax*10)

    # ones_lb.Scale(SetMax)  # scale them so the boxes are big enough
    # draw stuff
    xax.SetTitle("x crystal index")
    yax.SetTitle("y crystal index")
    if opt & 0x1:
        histo.Draw("col")
    else:
        histo.Draw("colz")

    ones_lb.Draw("boxsame")
    return ones_lb


def SetStyle(choice=0):
    if "R" not in globals():
        import ROOT as R
    if "gROOT" not in globals():
        from ROOT import gROOT

    hpsStyle = R.TStyle("HPS", "HPS style")

    # use plain black on white colors
    icol = 0
    hpsStyle.SetFrameBorderMode(icol)
    hpsStyle.SetCanvasBorderMode(icol)
    hpsStyle.SetPadBorderMode(icol)
    hpsStyle.SetPadColor(icol)
    hpsStyle.SetCanvasColor(icol)
    hpsStyle.SetStatColor(icol)
    # hpsStyle.SetFillColor(icol)

    # set the paper & margin sizes
    hpsStyle.SetPaperSize(20,26)
    hpsStyle.SetPadTopMargin(0.05)
    hpsStyle.SetPadRightMargin(0.05)
    hpsStyle.SetPadBottomMargin(0.18)
    hpsStyle.SetPadLeftMargin(0.14)

    # use large fonts
    # font=72
    if(choice == 0):
        font = 42         # helvetica-medium-r-normal
        title_font = 42
        title_size = 0.08
        title_size_z = 0.045

        label_font = 42
        label_size = 0.08
        label_size_z = 0.045
        hpsStyle.SetOptTitle(0)
        hpsStyle.SetOptStat(0)
        hpsStyle.SetOptFit(0)

    elif(choice == 1):
        #font=72
        font = 132        # times-medium-r-normal
        title_font = 132
        title_size = 0.08
        title_size_z = 0.045

        label_font = 42
        label_size = 0.035
        label_size_z = 0.035
        hpsStyle.SetOptTitle(0)
        hpsStyle.SetOptStat(0)
        hpsStyle.SetOptFit(0)


    hpsStyle.SetTextFont(font)
    hpsStyle.SetTextSize(label_size)

    hpsStyle.SetLabelFont(label_font, "x")
    hpsStyle.SetTitleFont(title_font, "x")
    hpsStyle.SetLabelFont(label_font, "y")
    hpsStyle.SetTitleFont(title_font, "y")
    hpsStyle.SetLabelFont(label_font, "z")
    hpsStyle.SetTitleFont(title_font, "z")

    hpsStyle.SetLabelSize(label_size, "x")
    hpsStyle.SetTitleSize(title_size, "x")
    hpsStyle.SetLabelSize(label_size, "y")
    hpsStyle.SetTitleSize(title_size, "y")
    hpsStyle.SetLabelSize(label_size_z, "z")
    hpsStyle.SetTitleSize(title_size_z, "z")

    hpsStyle.SetTitleOffset(0.7, "y")
    hpsStyle.SetTitleOffset(1.15, "x")

    #use bold lines and markers
    #hpsStyle.SetMarkerStyle(20)
    hpsStyle.SetMarkerSize(1.0)
    hpsStyle.SetHistLineWidth(1)
    hpsStyle.SetLineStyleString(2, "[12 12]")  # postscript dashes

    #get rid of X error bars and y error bar caps
    #hpsStyle.SetErrorX(0.001)

    #do not display any of the standard histogram decorations

    # put tick marks on top and RHS of plots
    hpsStyle.SetPadTickX(1)
    hpsStyle.SetPadTickY(1)

    #gROOT.SetStyle("Plain")
    gROOT.SetStyle("HPS")
    gROOT.ForceStyle()

    NCont = 255
    stops = R.std.vector("double")([0.00, 0.34, 0.61, 0.84, 1.00])
    red = R.std.vector("double")([0.00, 0.00, 0.87, 1.00, 0.51])
    green = R.std.vector("double")([0.00, 0.81, 1.00, 0.20, 0.00])
    blue = R.std.vector("double")([0.51, 1.00, 0.12, 0.00, 0.00])
    R.TColor.CreateGradientColorTable(stops.size(), stops.data(), red.data(), green.data(), blue.data(), NCont)
    R.gStyle.SetNumberContours(NCont)

/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/config-store-module.h"
#include "ns3/mygym.h"
// #include "ns3/netanim-module.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <iomanip>
#include "ns3/netanim-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LenaX2HandoverMeasures");

uint32_t RunNum;

/**
 * Sample simulation script for an automatic X2-based handover based on the RSRQ measures.
 * It instantiates two eNodeB, attaches one UE to the 'source' eNB.
 * The UE moves between both eNBs, it reports measures to the serving eNB and
 * the 'source' (serving) eNB triggers the handover of the UE towards
 * the 'target' eNB when it considers it is a better eNB.
 */
int
main (int argc, char *argv[])
{

  uint16_t numberOfUes = 90;
  uint16_t numberOfEnbs = 5;
  uint16_t numBearersPerUe = 5;
  bool useUdp = true;
  bool epcDl = true;
  bool epcUl = true;
  double speed = 20;   // m/s
  double simTime = 60; 
  double enbTxPowerDbm = 32.0; 
  double steptime = 0.5;
  uint16_t macroEnbBandwidth = 100;  // Valid BW options = {6, 15, 25, 50, 75, 100}
  uint16_t macroEnbDlEarfcn = 100;  
  uint16_t srsPeriodicity = 80;
  
  //opengym environment
  uint32_t openGymPort = 1150;

  std::string m_traceFile;

  // change some default attributes so that they are reasonable for
  // this scenario, but do this before processing command line
  // arguments, so that the user is allowed to override these settings
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (true));
  Config::SetDefault ("ns3::LteHelper::Scheduler", StringValue ("ns3::RrFfMacScheduler"));
  Config::SetDefault ("ns3::UdpClient::Interval", TimeValue (MilliSeconds (1)));
  Config::SetDefault ("ns3::UdpClient::PacketSize", UintegerValue(12)); // bytes
  Config::SetDefault ("ns3::UdpClient::MaxPackets", UintegerValue (1000000));
  Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (10 * 1024));
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (enbTxPowerDbm));
  Config::SetDefault("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue(srsPeriodicity));

  // Command line arguments
  CommandLine cmd;
  cmd.AddValue ("simTime", "Total duration of the simulation (in seconds)", simTime);
  cmd.AddValue ("speed", "Speed of the UE (default = 20 m/s)", speed);
  cmd.AddValue ("enbTxPowerDbm", "TX power [dBm] used by HeNBs (default = 46.0)", enbTxPowerDbm);
  cmd.AddValue ("RunNum" , "1...10" , RunNum);
  cmd.AddValue ("Bandwidth", "Bandwidth for eNB", macroEnbBandwidth);
  cmd.AddValue ("Bearer", "Number of bearer per UE", numBearersPerUe);
  cmd.AddValue ("SrsPeriodicity", "The SRS periodicity in milliseconds", srsPeriodicity);

  cmd.Parse (argc, argv);

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  // lteHelper->SetSchedulerType ("ns3::RrFfMacScheduler");
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");

  lteHelper->SetAttribute("PathlossModel", StringValue("ns3::Cost231PropagationLossModel"));
  lteHelper->SetSpectrumChannelType("ns3::MultiModelSpectrumChannel");
  
  //Test (eNB badwidth)
  lteHelper->SetEnbAntennaModelType ("ns3::IsotropicAntennaModel");
  lteHelper -> SetEnbDeviceAttribute("DlEarfcn", UintegerValue(macroEnbDlEarfcn));
  lteHelper -> SetEnbDeviceAttribute("UlEarfcn", UintegerValue(macroEnbDlEarfcn + 18000));
  lteHelper -> SetEnbDeviceAttribute("DlBandwidth", UintegerValue(macroEnbBandwidth));
  lteHelper -> SetEnbDeviceAttribute("UlBandwidth", UintegerValue(macroEnbBandwidth));

   lteHelper->SetHandoverAlgorithmType ("ns3::A3RsrpHandoverAlgorithm");
   lteHelper->SetHandoverAlgorithmAttribute ("Hysteresis",
                                             DoubleValue (0.0));
   lteHelper->SetHandoverAlgorithmAttribute ("TimeToTrigger",
                                             TimeValue (MilliSeconds (0)));

  Ptr<Node> pgw = epcHelper->GetPgwNode ();

  // Create a single RemoteHost
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  Ptr<Node> remoteHost = remoteHostContainer.Get (0);
  InternetStackHelper internet;
  internet.Install (remoteHostContainer);

  // Create the Internet
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Mb/s")));
  p2ph.SetDeviceAttribute ("Mtu", UintegerValue (1500));
  p2ph.SetChannelAttribute ("Delay", TimeValue (Seconds (0.010)));
  NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
  Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);

  // Routing of the Internet Host (towards the LTE network)
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  // interface 0 is localhost, 1 is the p2p device
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);


  NodeContainer ueNodes;
  NodeContainer dummy_enbNodes;
  NodeContainer enbNodes;
  enbNodes.Create (numberOfEnbs-2);
  ueNodes.Create (numberOfUes);
  dummy_enbNodes.Create (2);


  // Install Mobility Model in eNB
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (10000, 10000, 10000));
  positionAlloc->Add (Vector (10000, 10000, 10000));

  MobilityHelper dummy_enb_mobility;
  dummy_enb_mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  dummy_enb_mobility.SetPositionAllocator (positionAlloc);
  dummy_enb_mobility.Install (dummy_enbNodes);
  
  MobilityHelper enbMobility;
  enbMobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                "MinX", DoubleValue (500.0),
                                "MinY", DoubleValue (500.0),
                                "DeltaX", DoubleValue (500.0),
                                "DeltaY", DoubleValue (500.0),
                                "GridWidth", UintegerValue (3),
                                "LayoutType", StringValue ("RowFirst"));
  enbMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  enbMobility.Install (enbNodes);

  // Install Mobility Model in UE
  MobilityHelper ueMobility;
  Ptr<RandomRectanglePositionAllocator> allocator = CreateObject<RandomRectanglePositionAllocator> ();
  Ptr<UniformRandomVariable> xPos = CreateObject<UniformRandomVariable> ();
  xPos->SetAttribute ("Min", DoubleValue (400.0));
  xPos->SetAttribute ("Max", DoubleValue (1600.0));
  allocator->SetX (xPos);
  Ptr<UniformRandomVariable> yPos = CreateObject<UniformRandomVariable> ();
  yPos->SetAttribute ("Min", DoubleValue (400.0));
  yPos->SetAttribute ("Max", DoubleValue (600.0));
  allocator->SetY (yPos);
  allocator->AssignStreams (1);
  ueMobility.SetPositionAllocator (allocator);
  ueMobility.SetMobilityModel ("ns3::RandomDirection2dMobilityModel",
                             "Bounds", RectangleValue (Rectangle (300, 1700, 300, 700)),
                             "Speed", StringValue ("ns3::ConstantRandomVariable[Constant=3]"),
                             "Pause", StringValue ("ns3::ConstantRandomVariable[Constant=0.1]"));
  ueMobility.Install (ueNodes);

  Ptr<MyGymEnv> son_server = CreateObject<MyGymEnv> (steptime, numberOfEnbs, numberOfUes, macroEnbBandwidth);
  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort);

  son_server->SetOpenGymInterface(openGymInterface);

  // Install LTE Devices in eNB and UEs
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (enbTxPowerDbm));
  lteHelper->SetEnbAntennaModelType ("ns3::IsotropicAntennaModel");
  lteHelper->SetEnbDeviceAttribute ("DlEarfcn", UintegerValue (macroEnbDlEarfcn));
  lteHelper->SetEnbDeviceAttribute ("UlEarfcn", UintegerValue (macroEnbDlEarfcn + 18000));
  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (macroEnbBandwidth));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (macroEnbBandwidth));
  NetDeviceContainer dummy_enb = lteHelper->InstallEnbDevice (dummy_enbNodes, son_server);
  NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice (enbNodes, son_server);
  NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice (ueNodes, son_server);

  // Install the IP stack on the UEs
  internet.Install (ueNodes);
  Ipv4InterfaceContainer ueIpIfaces;
  ueIpIfaces = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueLteDevs));

  // Attach all UEs to the first eNodeB
  for (uint16_t i = 0; i < numberOfUes; i++)
    {
      lteHelper->AttachToClosestEnb (ueLteDevs.Get(i), enbLteDevs);
    }


  NS_LOG_LOGIC ("setting up applications");

  // Install and start applications on UEs and remote host
  uint16_t dlPort = 10000;
  uint16_t ulPort = 20000;

  // randomize a bit start times to avoid simulation artifacts
  // (e.g., buffer overflows due to packet transmissions happening
  // exactly at the same time)
  Ptr<UniformRandomVariable> startTimeSeconds = CreateObject<UniformRandomVariable> ();
  startTimeSeconds->SetAttribute ("Min", DoubleValue (0));
  startTimeSeconds->SetAttribute ("Max", DoubleValue (0.010));

  for (uint32_t u = 0; u < numberOfUes; ++u)
    {
      Ptr<Node> ue = ueNodes.Get (u);
            // Set the default gateway for the UE
            Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ue->GetObject<Ipv4> ());
            ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);

            for (uint32_t b = 0; b < numBearersPerUe; ++b)
              {
                ++dlPort;
                ++ulPort;

                ApplicationContainer clientApps;
                ApplicationContainer serverApps;

                if (useUdp)
                  {
                    if (epcDl)
                      {
                        NS_LOG_UNCOND ("installing UDP DL app for UE " << u);
                        UdpClientHelper dlClientHelper (ueIpIfaces.GetAddress (u), dlPort);
                        clientApps.Add (dlClientHelper.Install (remoteHost));
                        PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory",
                                                                 InetSocketAddress (Ipv4Address::GetAny (), dlPort));
                        serverApps.Add (dlPacketSinkHelper.Install (ue));
                      }
                    if (epcUl)
                      {
                        NS_LOG_UNCOND ("installing UDP UL app for UE " << u);
                        UdpClientHelper ulClientHelper (remoteHostAddr, ulPort);
                        clientApps.Add (ulClientHelper.Install (ue));
                        PacketSinkHelper ulPacketSinkHelper ("ns3::UdpSocketFactory",
                                                             InetSocketAddress (Ipv4Address::GetAny (), ulPort));
                          serverApps.Add (ulPacketSinkHelper.Install (remoteHost));
                      }
                  }
                else // use TCP
                  {
                    if (epcDl)
                      {
                        NS_LOG_UNCOND ("installing TCP DL app for UE " << u);
                        BulkSendHelper dlClientHelper ("ns3::TcpSocketFactory",
                                                       InetSocketAddress (ueIpIfaces.GetAddress (u), dlPort));
                        dlClientHelper.SetAttribute ("MaxBytes", UintegerValue (0));
                        clientApps.Add (dlClientHelper.Install (remoteHost));
                        PacketSinkHelper dlPacketSinkHelper ("ns3::TcpSocketFactory",
                                                             InetSocketAddress (Ipv4Address::GetAny (), dlPort));
                        serverApps.Add (dlPacketSinkHelper.Install (ue));
                      }
                    if (epcUl)
                      {
                        NS_LOG_UNCOND ("installing TCP UL app for UE " << u);
                        BulkSendHelper ulClientHelper ("ns3::TcpSocketFactory",
                                                       InetSocketAddress (remoteHostAddr, ulPort));
                        ulClientHelper.SetAttribute ("MaxBytes", UintegerValue (0));
                        clientApps.Add (ulClientHelper.Install (ue));
                        PacketSinkHelper ulPacketSinkHelper ("ns3::TcpSocketFactory",
                                                             InetSocketAddress (Ipv4Address::GetAny (), ulPort));
                        serverApps.Add (ulPacketSinkHelper.Install (remoteHost));
                      }
                  } // end if (useUdp)

                Ptr<EpcTft> tft = Create<EpcTft> ();
                if (epcDl)
                  {
                    EpcTft::PacketFilter dlpf;
                    dlpf.localPortStart = dlPort;
                    dlpf.localPortEnd = dlPort;
                    tft->Add (dlpf);
                  }
                if (epcUl)
                  {
                    EpcTft::PacketFilter ulpf;
                    ulpf.remotePortStart = ulPort;
                    ulpf.remotePortEnd = ulPort;
                    tft->Add (ulpf);
                  }

                if (epcDl || epcUl)
                  {
                    EpsBearer bearer (EpsBearer::NGBR_VIDEO_TCP_DEFAULT);
                    lteHelper->ActivateDedicatedEpsBearer (ueLteDevs.Get (u), bearer, tft);
                  }
                Time startTime = Seconds (startTimeSeconds->GetValue ());
                serverApps.Start (startTime);
                clientApps.Start (startTime);

              } // end for b
          }


  // Add X2 interface
  lteHelper->AddX2Interface (enbNodes);

  // lteHelper->EnablePhyTraces ();
  // lteHelper->EnableMacTraces ();
  lteHelper->EnableRlcTraces ();
  // lteHelper->EnablePdcpTraces ();
  Ptr<RadioBearerStatsCalculator> rlcStats = lteHelper->GetRlcStats (son_server);
  rlcStats->SetAttribute ("EpochDuration", TimeValue (Seconds (0.5)));

  // For results
  rlcStats->SetAttribute ("DlRlcOutputFilename", StringValue ("DlRlcStats.txt"));

  for (uint32_t it = 0; it != enbNodes.GetN(); ++it) {
        Ptr < NetDevice > netDevice = enbLteDevs.Get(it);
        Ptr < LteEnbNetDevice > enbNetDevice = netDevice -> GetObject < LteEnbNetDevice > ();
        Ptr < LteEnbPhy > enbPhy = enbNetDevice -> GetPhy();
        enbPhy -> TraceConnectWithoutContext("DlPhyTransmission", MakeBoundCallback( & MyGymEnv::GetPhyStats, son_server));
    }

  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();
  Simulator::Destroy ();
  return 0;

}
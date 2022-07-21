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


void
NotifyConnectionEstablishedUe (std::string context,
                               uint64_t imsi,
                               uint16_t cellid,
                               uint16_t rnti)
{
  std::cout << context
            << " UE IMSI " << imsi
            << ": connected to CellId " << cellid
            << " with RNTI " << rnti
            << std::endl;
}

void
NotifyHandoverStartUe (std::string context,
                       uint64_t imsi,
                       uint16_t cellid,
                       uint16_t rnti,
                       uint16_t targetCellId)
{
  std::cout << context
            << " UE IMSI " << imsi
            << ": previously connected to CellId " << cellid
            << " with RNTI " << rnti
            << ", doing handover to CellId " << targetCellId
            << std::endl;
}

void
NotifyHandoverEndOkUe (std::string context,
                       uint64_t imsi,
                       uint16_t cellid,
                       uint16_t rnti)
{
  std::cout << context
            << " UE IMSI " << imsi
            << ": successful handover to CellId " << cellid
            << " with RNTI " << rnti
            << std::endl;
}

void
NotifyConnectionEstablishedEnb (std::string context,
                                uint64_t imsi,
                                uint16_t cellid,
                                uint16_t rnti)
{
  std::cout << context
            << " eNB CellId " << cellid
            << ": successful connection of UE with IMSI " << imsi
            << " RNTI " << rnti
            << std::endl;
}

void
NotifyHandoverStartEnb (std::string context,
                        uint64_t imsi,
                        uint16_t cellid,
                        uint16_t rnti,
                        uint16_t targetCellId)
{
  std::cout << context
            << " eNB CellId " << cellid
            << ": start handover of UE with IMSI " << imsi
            << " RNTI " << rnti
            << " to CellId " << targetCellId
            << std::endl;
}

void
NotifyHandoverEndOkEnb (std::string context,
                        uint64_t imsi,
                        uint16_t cellid,
                        uint16_t rnti)
{
  std::cout << context
            << " eNB CellId " << cellid
            << ": completed handover of UE with IMSI " << imsi
            << " RNTI " << rnti
            << std::endl;
}


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
  // LogLevel logLevel = (LogLevel)(LOG_PREFIX_ALL | LOG_LEVEL_ALL);

  // LogComponentEnable ("LteHelper", logLevel);
  // LogComponentEnable ("EpcHelper", logLevel);
  // LogComponentEnable ("EpcEnbApplication", logLevel);
  // LogComponentEnable ("EpcMmeApplication", logLevel);
  // LogComponentEnable ("EpcPgwApplication", logLevel);
  // LogComponentEnable ("EpcSgwApplication", logLevel);
  // LogComponentEnable ("EpcX2", logLevel);

  // LogComponentEnable ("RrFfMacScheduler", logLevel);
  // LogComponentEnable ("LteEnbRrc", logLevel);
  // LogComponentEnable ("LteEnbNetDevice", logLevel);
  // LogComponentEnable ("LteUeRrc", logLevel);
  // LogComponentEnable ("LteUeNetDevice", logLevel);
  // LogComponentEnable ("A2A4RsrqHandoverAlgorithm", logLevel);
  // LogComponentEnable ("A3RsrpHandoverAlgorithm", logLevel);

  uint16_t numberOfUes = 20;
  uint16_t numberOfEnbs = 3;
  uint16_t numBearersPerUe = 1;
  bool disableDl = false;
  bool disableUl = false;
  double distance = 100.0; // m
  // double yForUe = 500.0;   // m
  double speed = 20;       // m/s
  double simTime = 10000 + (double)(numberOfEnbs + 1) * distance / speed; // 1500 m / 20 m/s = 75 secs
  double enbTxPowerDbm = 46.0;
  double steptime = 0.5;
  // Valid BW options = {6, 15, 25, 50, 75, 100}
  // uint16_t macroEnbBandwidth = 50;
  uint16_t macroEnbBandwidth = 75;

  //opengym environment
  uint32_t openGymPort = 1131;

  // change some default attributes so that they are reasonable for
  // this scenario, but do this before processing command line
  // arguments, so that the user is allowed to override these settings
  Config::SetDefault ("ns3::UdpClient::Interval", TimeValue (MilliSeconds (10)));
  Config::SetDefault ("ns3::UdpClient::MaxPackets", UintegerValue (100000));
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (true));

  // Command line arguments
  CommandLine cmd;
  cmd.AddValue ("simTime", "Total duration of the simulation (in seconds)", simTime);
  cmd.AddValue ("speed", "Speed of the UE (default = 20 m/s)", speed);
  cmd.AddValue ("enbTxPowerDbm", "TX power [dBm] used by HeNBs (default = 46.0)", enbTxPowerDbm);

  cmd.Parse (argc, argv);


  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);
  // lteHelper->SetSchedulerType ("ns3::RrFfMacScheduler");
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");

  //Test (eNB badwidth)
  lteHelper -> SetEnbDeviceAttribute("DlBandwidth", UintegerValue(macroEnbBandwidth));
  lteHelper -> SetEnbDeviceAttribute("UlBandwidth", UintegerValue(macroEnbBandwidth));

  // lteHelper->SetHandoverAlgorithmType ("ns3::A2A4RsrqHandoverAlgorithm");
  // lteHelper->SetHandoverAlgorithmAttribute ("ServingCellThreshold",
  //                                           UintegerValue (30));
  // lteHelper->SetHandoverAlgorithmAttribute ("NeighbourCellOffset",
  //                                           UintegerValue (1));

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
  // To Check Remote Host's Address
  // std::cout << "Remote Host Address" << internetIpIfaces.GetAddress(1);


  // Routing of the Internet Host (towards the LTE network)
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  // interface 0 is localhost, 1 is the p2p device
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);

  /*
   * Network topology:
   *
   *      |     + --------------------------------------------------------->
   *      |     UE
   *      |
   *      |               d                   d                   d
   *    y |     |-------------------x-------------------x-------------------
   *      |     |                 eNodeB              eNodeB
   *      |   d |
   *      |     |
   *      |     |                                             d = distance
   *            o (0, 0, 0)                                   y = yForUe
   */

  NodeContainer ueNodes;
  NodeContainer enbNodes;
  enbNodes.Create (numberOfEnbs);
  ueNodes.Create (numberOfUes);
  
  // Install Mobility Model
  // Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator> ();



  // enbPositionAlloc->Add (Vector (0, 0, 0));
  // enbPositionAlloc->Add (Vector (100, 0, 0));
  // enbPositionAlloc->Add (Vector (100, 100, 0));
  // enbPositionAlloc->Add (Vector (0, 100, 0));
  // // positionAlloc->Add (Vector (100/2, 100/2, 0));
  // MobilityHelper enbMobility;
  // enbMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  // enbMobility.SetPositionAllocator (enbPositionAlloc);
  // enbMobility.Install (enbNodes);
  // mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  // mobility.SetPositionAllocator (positionAlloc);
  // mobility.Install (enbNodes);
  // mobility.Install (ueNodes);


  // Install Mobility Model in eNB
  MobilityHelper enbMobility;
  enbMobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                "MinX", DoubleValue (100.0),
                                "MinY", DoubleValue (100.0),
                                "DeltaX", DoubleValue (100.0),
                                "DeltaY", DoubleValue (100.0),
                                "GridWidth", UintegerValue (3),
                                "LayoutType", StringValue ("RowFirst"));
  enbMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  enbMobility.Install (enbNodes);

  //To Check eNB's Position
  //for (uint32_t e = 0; e < numberOfEnbs; ++e)
  //  {
  //    Ptr<Node> eNb = enbNodes.Get (e);
  //    Ptr<MobilityModel> mob = eNb->GetObject<MobilityModel>();
  //    std::cout << "eNBPostion.X" << mob->GetPosition().x <<  "eNBPostion.Y" << mob->GetPosition().y << std::endl;
  //  }

  // Install Mobility Model in UE
  MobilityHelper ueMobility;
  Ptr<RandomRectanglePositionAllocator> allocator = CreateObject<RandomRectanglePositionAllocator> ();
  Ptr<UniformRandomVariable> xPos = CreateObject<UniformRandomVariable> ();
  xPos->SetAttribute ("Min", DoubleValue (0.0));
  xPos->SetAttribute ("Max", DoubleValue (400.0));
  allocator->SetX (xPos);
  Ptr<UniformRandomVariable> yPos = CreateObject<UniformRandomVariable> ();
  yPos->SetAttribute ("Min", DoubleValue (0.0));
  yPos->SetAttribute ("Max", DoubleValue (200.0));
  allocator->SetY (yPos);
  allocator->AssignStreams (1);
  ueMobility.SetPositionAllocator (allocator);
  ueMobility.SetMobilityModel ("ns3::RandomDirection2dMobilityModel",
                             "Bounds", RectangleValue (Rectangle (0, 400, 0, 200)),
                             "Speed", StringValue ("ns3::ConstantRandomVariable[Constant=5]"),
                             "Pause", StringValue ("ns3::ConstantRandomVariable[Constant=0.1]"));
  ueMobility.Install (ueNodes);

  //To Check UE's Position
  //for (uint32_t u = 0; u < numberOfUes; ++u)
  //  {
  //    Ptr<Node> ue = ueNodes.Get (u);
  //    Ptr<MobilityModel> mob = ue->GetObject<MobilityModel>();
  //    std::cout << "UEPostion.X" << mob->GetPosition().x <<  "UEPostion.Y" << mob->GetPosition().y << std::endl;
  //  }

  // ueMobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
  //                           "Mode", StringValue ("Time"),
  //                           "Time", StringValue ("1s"),
  //                           "Speed", StringValue ("ns3::ConstantRandomVariable[Constant=5.0]"),
  //                           "Bounds", StringValue ("0|400|0|300"));
  // ueMobility.Install (ueNodes);


  // // Install Mobility Model in UE
  // MobilityHelper ueMobility;
  // ueMobility.SetPositionAllocator ("ns3::GridPositionAllocator",
  //                                "MinX", DoubleValue (100.0),
  //                                "MinY", DoubleValue (100.0),
  //                                "DeltaX", DoubleValue (20.0),
  //                                "DeltaY", DoubleValue (20.0),
  //                                "GridWidth", UintegerValue (5),
  //                                "LayoutType", StringValue ("RowFirst"));
  // ueMobility.SetMobilityModel ("ns3::RandomDirection2dMobilityModel",
  //                            "Bounds", RectangleValue (Rectangle (-500, 500, -500, 500)),
  //                            "Speed", StringValue ("ns3::ConstantRandomVariable[Constant=2]"),
  //                            "Pause", StringValue ("ns3::ConstantRandomVariable[Constant=0.2]"));
  // ueMobility.Install (ueNodes);

  Ptr<MyGymEnv> son_server = CreateObject<MyGymEnv> (steptime, numberOfEnbs, numberOfUes, macroEnbBandwidth);
  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort);

  son_server->SetOpenGymInterface(openGymInterface);

  // Install LTE Devices in eNB and UEs
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (enbTxPowerDbm));
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

  DataRateValue dataRateValue = DataRate ("100Mbps");

  // No use
  uint64_t bitRate = dataRateValue.Get ().GetBitRate ();

  // uint32_t packetSize = 1024; //bytes

  NS_LOG_DEBUG ("bit rate " << bitRate);

  // double interPacketInterval = static_cast<double> (packetSize * 8) / bitRate;

  // Time udpInterval = Seconds (interPacketInterval);
  // No use
  Time udpInterval = Seconds (0.01);

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
          ApplicationContainer clientApps;
          ApplicationContainer serverApps;
          Ptr<EpcTft> tft = Create<EpcTft> ();

          if (!disableDl)
            {
              ++dlPort;

              NS_LOG_LOGIC ("installing UDP DL app for UE " << u);
              UdpClientHelper dlClientHelper (ueIpIfaces.GetAddress (u), dlPort);
              // dlClientHelper.SetAttribute ("Interval", TimeValue (udpInterval));
              // dlClientHelper.SetAttribute ("PacketSize", UintegerValue (packetSize));
              // dlClientHelper.SetAttribute ("MaxPackets", UintegerValue (100));
              clientApps.Add (dlClientHelper.Install (remoteHost));
              PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory",
                                                   InetSocketAddress (Ipv4Address::GetAny (), dlPort));
              serverApps.Add (dlPacketSinkHelper.Install (ue));

              EpcTft::PacketFilter dlpf;
              dlpf.localPortStart = dlPort;
              dlpf.localPortEnd = dlPort;
              tft->Add (dlpf);
            }

          if (!disableUl)
            {
              ++ulPort;

              NS_LOG_LOGIC ("installing UDP UL app for UE " << u);
              UdpClientHelper ulClientHelper (remoteHostAddr, ulPort);
              // ulClientHelper.SetAttribute ("Interval", TimeValue (udpInterval));
              // ulClientHelper.SetAttribute ("PacketSize", UintegerValue (packetSize));
              // ulClientHelper.SetAttribute ("MaxPackets", UintegerValue (100));
              clientApps.Add (ulClientHelper.Install (ue));
              PacketSinkHelper ulPacketSinkHelper ("ns3::UdpSocketFactory",
                                                   InetSocketAddress (Ipv4Address::GetAny (), ulPort));
              serverApps.Add (ulPacketSinkHelper.Install (remoteHost));

              EpcTft::PacketFilter ulpf;
              ulpf.remotePortStart = ulPort;
              ulpf.remotePortEnd = ulPort;
              tft->Add (ulpf);
            }

          EpsBearer bearer (EpsBearer::NGBR_VIDEO_TCP_DEFAULT);
          lteHelper->ActivateDedicatedEpsBearer (ueLteDevs.Get (u), bearer, tft);

          Time startTime = Seconds (startTimeSeconds->GetValue ());
          serverApps.Start (startTime);
          clientApps.Start (startTime);

        } // end for b
    }


  // Add X2 interface
  lteHelper->AddX2Interface (enbNodes);

  // X2-based Handover
  //lteHelper->HandoverRequest (Seconds (0.100), ueLteDevs.Get (0), enbLteDevs.Get (0), enbLteDevs.Get (1));

  // Uncomment to enable PCAP tracing
  // p2ph.EnablePcapAll("lena-x2-handover-measures");

  lteHelper->EnablePhyTraces ();
  // lteHelper->EnableMacTraces ();
  lteHelper->EnableRlcTraces ();
  lteHelper->EnablePdcpTraces ();
  Ptr<RadioBearerStatsCalculator> rlcStats = lteHelper->GetRlcStats ();
  rlcStats->SetAttribute ("EpochDuration", TimeValue (Seconds (1.0)));

  // Non-algorithm case -> Go to "test.py and configure"
  // rlcStats->SetAttribute ("DlRlcOutputFilename", StringValue ("DlRlcStats_random_non_alg.txt"));
  
  // // MLB Algorithm case -> Go to "test.py and configure"
  // rlcStats->SetAttribute ("DlRlcOutputFilename", StringValue ("DlRlcStats_random_alg.txt"));

  Ptr<RadioBearerStatsCalculator> pdcpStats = lteHelper->GetPdcpStats ();
  pdcpStats->SetAttribute ("EpochDuration", TimeValue (Seconds (1.0)));

  for (uint32_t it = 0; it != enbNodes.GetN(); ++it) {
        Ptr < NetDevice > netDevice = enbLteDevs.Get(it);
        Ptr < LteEnbNetDevice > enbNetDevice = netDevice -> GetObject < LteEnbNetDevice > ();
        Ptr < LteEnbPhy > enbPhy = enbNetDevice -> GetPhy();
        enbPhy -> TraceConnectWithoutContext("DlPhyTransmission", MakeBoundCallback( & MyGymEnv::GetPhyStats, son_server));
    }

  // connect custom trace sinks for RRC connection establishment and handover notification
  // Config::Connect ("/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
  //                  MakeCallback (&NotifyConnectionEstablishedEnb));
  // Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
  //                  MakeCallback (&NotifyConnectionEstablishedUe));
  // Config::Connect ("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
  //                  MakeCallback (&NotifyHandoverStartEnb));
  // Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/HandoverStart",
  //                  MakeCallback (&NotifyHandoverStartUe));
  // Config::Connect ("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
  //                  MakeCallback (&NotifyHandoverEndOkEnb));
  // Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/HandoverEndOk",
  //                  MakeCallback (&NotifyHandoverEndOkUe));


  Simulator::Stop (Seconds (simTime));

  //To Reduce xml file size
  // AnimationInterface anim ("mlb_alg.xml"); 
  // anim.SetMaxPktsPerTraceFile(50000);
  // anim.SetMobilityPollInterval(Seconds(0.3)); //Standard : 0.25(s), UE Moving interval : 0.2(s)
  // anim.EnablePacketMetadata (false); //Not include additional information

  Simulator::Run ();

  // GtkConfigStore config;
  // config.ConfigureAttributes ();

  // openGymInterface->NotifySimulationEnd();
  Simulator::Destroy ();
  return 0;

}
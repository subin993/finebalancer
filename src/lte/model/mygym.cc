/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Technische Universität Berlin
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
 */

#include "mygym.h"

#include "ns3/object.h"

#include "ns3/core-module.h"

#include <ns3/lte-module.h>

#include "ns3/node-list.h"

#include "ns3/log.h"

#include <sstream>

#include <iostream>

#include <ns3/lte-common.h>
#include "ns3/lte-enb-rrc.h"

#include <ns3/cell-individual-offset.h>
// #include <ns3/radio-bearer-stats-calculator.h>

#include <stdlib.h>

#include <typeinfo>

#include <numeric>

#include <cmath>

namespace ns3 {

    NS_LOG_COMPONENT_DEFINE("MyGymEnv");

    NS_OBJECT_ENSURE_REGISTERED(MyGymEnv);

    MyGymEnv::MyGymEnv() {
        NS_LOG_FUNCTION(this);
    }

    MyGymEnv::MyGymEnv(double stepTime, uint32_t N1, uint32_t N2, uint16_t N3) {
        NS_LOG_FUNCTION(this);

        //
        m_step = 0.0;
        collect = 0;
        collecting_window = 0.05; //50ms
        block_Thr = 0.5; // Blockage threshold 0.5 Mb/s
        m_interval = stepTime;
        m_cellCount = N1;
        m_userCount = N2;
        m_nRBTotal = N3 * collecting_window * 1000;
        m_rbUtil.assign(m_cellCount, 0);
        m_dlThroughput = 0;
        m_cellFrequency.assign(m_cellCount, 0);
        rewards.assign(m_cellCount, 0);
        rewards_sum.assign(m_cellCount, 0);
        m_dlThroughputVec.assign(m_cellCount, 0);
        m_UesNum.assign(m_cellCount, 0);
        std::vector < uint32_t > dummyVec(29, 0);
        m_MCSPen.assign(m_cellCount, dummyVec);
        Simulator::Schedule(Seconds(0), & MyGymEnv::ScheduleNextStateRead, this);
        Simulator::Schedule(Seconds(m_interval - collecting_window), & MyGymEnv::Start_Collecting, this);
        UserThrouput.clear();
    }

    void
    MyGymEnv::ScheduleNextStateRead() {
        
        m_step ++;
        NS_LOG_FUNCTION(this);
        Simulator::Schedule(Seconds(m_interval), & MyGymEnv::ScheduleNextStateRead, this);
        Notify();
    }
    void
    MyGymEnv::Start_Collecting() {
        NS_LOG_FUNCTION(this);
        collect = 1;
        NS_LOG_LOGIC("%%%%%%%% Start collecting %%%%%%%%  time= " << Simulator::Now().GetSeconds() << " sec");

    }
    MyGymEnv::~MyGymEnv() {
        NS_LOG_FUNCTION(this);
    }

    TypeId
    MyGymEnv::GetTypeId(void) {
        static TypeId tid = TypeId("MyGymEnv")
            .SetParent < OpenGymEnv > ()
            .SetGroupName("OpenGym")
            .AddConstructor < MyGymEnv > ();
        return tid;
    }

    void
    MyGymEnv::DoDispose() {
        NS_LOG_FUNCTION(this);
    }

    // NS-3 SON
    void 
    MyGymEnv::GetRlcStats(Ptr<RadioBearerStatsCalculator> m_rlcStats) {
        RlcStats = m_rlcStats;
    }
    
    void 
    MyGymEnv::AddNewNode(uint16_t cellId, Ptr<LteEnbNetDevice> dev){
        m_enbs.insert(std::pair<uint32_t, Ptr<LteEnbNetDevice>> (cellId, dev));
    }
    void 
    MyGymEnv::AddNewUe(uint64_t imsi, Ptr<LteUeNetDevice> dev){
        m_ues.insert(std::pair<uint64_t, Ptr<LteUeNetDevice>> (imsi, dev));
    }
    
    Ptr < OpenGymSpace >
        MyGymEnv::GetActionSpace() {
            
            uint32_t nodeNum = m_enbs.size();

            std::vector<uint32_t> shape {nodeNum,};
            std::string dtype = TypeNameGet<float> ();

            Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (-6, 6, shape, dtype); //todo range change for -30 to 30
            
            NS_LOG_INFO("GetActionSpace: "<<space);

            return space;
        }

    Ptr < OpenGymSpace >
        MyGymEnv::GetObservationSpace() {
            NS_LOG_FUNCTION(this);

            // Original (Ghada-sy)
            // RB Util, DL Throughput, MCS ratio
            ///////////////////////////
            Ptr < OpenGymDictSpace > space = CreateObject < OpenGymDictSpace > ();

            std::vector < uint32_t > shape = { m_cellCount,};
            std::string dtype = TypeNameGet < float > ();

            // RB Util
            float low = 0.0;
            float high = 1.0;
            Ptr < OpenGymBoxSpace > rbUtil = CreateObject < OpenGymBoxSpace > (low, high, shape, dtype);
            bool flag = space -> Add("rbUtil", rbUtil);
            if (flag) {
                NS_LOG_LOGIC("rbUtil added to dictionary");
            } else {
                NS_LOG_LOGIC("Failed to add rbUtil to dictionary");
            }

            // DL throughput from Ghada-sy
            low = 0.0;
            high = 0.009422 * m_interval * 1000;
            Ptr < OpenGymBoxSpace > dlThroughput = CreateObject < OpenGymBoxSpace > (low, high, shape, dtype);
            flag = space -> Add("dlThroughput", dlThroughput);
            if (flag) {
                NS_LOG_LOGIC("rbUtil added to dictionary");
            } else {
                NS_LOG_LOGIC("Failed to add rbUtil to dictionary");
            }

            // MCS ratio
            low = 0.0;
            high = 1.0;
            shape = {  29, }; // LTE has 29 different MCS

            Ptr < OpenGymTupleSpace > MCSPen = CreateObject < OpenGymTupleSpace > ();
            for (uint8_t idx = 0; idx < m_cellCount; ++idx) {
                Ptr < OpenGymBoxSpace > MCSPenRow = CreateObject < OpenGymBoxSpace > (low, high, shape, dtype);
                flag = MCSPen -> Add(MCSPenRow);
            }

            flag = space -> Add("MCSPen", MCSPen);
            if (flag) {
                NS_LOG_LOGIC("MCSPen added to dictionary");
            } else {
                NS_LOG_LOGIC("MCSPen to add rbUtil to dictionary");
            }
            ///////////////////////////
            
            // Custom
            // FarUes, Served_UEs, AvgCQI, DL Throughput(from calculator), TotalCqi
            ///////////////////////////
            shape = { m_cellCount,};
            Ptr < OpenGymBoxSpace > FarUes = CreateObject < OpenGymBoxSpace > (low, high, shape, dtype);
            Ptr < OpenGymBoxSpace > ServedUes = CreateObject < OpenGymBoxSpace > (0, 40, shape, dtype);
            Ptr < OpenGymBoxSpace > AvgCqi = CreateObject < OpenGymBoxSpace > (0, 15, shape, dtype);
            Ptr < OpenGymBoxSpace > Throughput = CreateObject < OpenGymBoxSpace > (0, 50, shape, dtype);
            Ptr < OpenGymBoxSpace > TotalCqi = CreateObject < OpenGymBoxSpace > (0, 50, shape, dtype);

            space -> Add("FarUes", FarUes);
            space -> Add("ServedUes", ServedUes);
            space -> Add("AvgCqi", AvgCqi);
            space -> Add("Throughput", Throughput);
            space -> Add("TotalCqi",TotalCqi);
            ///////////////////////////

            return space;
        }

    bool
    MyGymEnv::GetGameOver() {
        NS_LOG_FUNCTION(this);
        bool isGameOver = false;
        NS_LOG_LOGIC("MyGetGameOver: " << isGameOver);
        return isGameOver;
    }

    Ptr < OpenGymDataContainer >
        MyGymEnv::GetObservation() {
            NS_LOG_FUNCTION(this);

            calculate_rewards();

            Ptr < OpenGymDictContainer > obsContainer = CreateObject < OpenGymDictContainer > ();
            std::vector < uint32_t > shape = { m_cellCount, };

            std::vector<float> normal_RB;
            std::vector<float> Far_UEs;
            std::vector<float> Served_UEs;
            std::vector<float> PRB_lin;
            std::vector<double> location_x;
            std::vector<double> location_y;

            uint32_t enbIdx = 1;
            for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter){
                std::cout<<"eNB "<<enbIdx<<": "<<iter->second->GetRrc ()->m_numofues<< " UEs " <<std::endl;
                enbIdx ++;
            }
            std::cout<<"**************************"<<std::endl;  

            // Original (Ghada-sy)
            // RB Util, DL Throughput, MCS ratio
            ///////////////////////////

            // RB Util
            Ptr < OpenGymBoxContainer < float > > box = CreateObject < OpenGymBoxContainer < float > > (shape);

            float normalizedRBUtil;
            for (uint8_t idx = 0; idx < m_cellCount; ++idx) {
                normalizedRBUtil = float(m_rbUtil.at(idx)) / m_nRBTotal ;

                normal_RB.push_back(normalizedRBUtil);
                
                box -> AddValue(normalizedRBUtil);
                
                //To see normalized RB util
                std::cout << "eNB "<< (idx+1) << ": " <<"Ghada-sy RB Utilization : "<<(float(m_rbUtil.at(idx))/m_nRBTotal)<<std::endl;
            }
            obsContainer -> Add("rbUtil", box);
            std::cout<<"**************************"<<std::endl;

            // Dl Throughput
            box = CreateObject < OpenGymBoxContainer < float > > (shape);
            box -> SetData(m_dlThroughputVec);
            obsContainer -> Add("dlThroughput", box);

            // MCS Ratio
            std::vector < float > dummyVec(29, 0.0);
            std::vector < std::vector < float >> normalizedMCSPen(m_cellCount, dummyVec);
            for (uint8_t idx = 0; idx < m_cellCount; ++idx) {
                for (uint8_t idx2 = 0; idx2 < 29; ++idx2) {
                    if (m_cellFrequency[idx] != 0) {
                        normalizedMCSPen[idx][idx2] = float(m_MCSPen[idx][idx2]) / m_cellFrequency[idx];
                    } else {
                        normalizedMCSPen[idx][idx2] = 0.0;
                    }
                }
            }
            shape = { 29, };
            Ptr < OpenGymTupleContainer > mcsPenContainer = CreateObject < OpenGymTupleContainer > ();
            for (uint8_t idx = 0; idx < m_cellCount; ++idx) {
                box = CreateObject < OpenGymBoxContainer < float > > (shape);
                box -> SetData(normalizedMCSPen.at(idx));
                mcsPenContainer -> Add(box);
            }
            obsContainer -> Add("MCSPen", mcsPenContainer);
            
            ///////////////////////////

            
            // Custom
            // FarUes, Served_UEs, AvgCQI, DL Throughput(from calculator), TotalCqi
            ///////////////////////////

            // Far UEs, Served UEs
            for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter){
                uint32_t CellId = iter->first;
                std::vector<uint64_t> Imsi_List;
                double eNB_x = iter->second->GetNode()->GetObject<MobilityModel>()->GetPosition().x;
                double eNB_y = iter->second->GetNode()->GetObject<MobilityModel>()->GetPosition().y;
                double eNB_z = iter->second->GetNode()->GetObject<MobilityModel>()->GetPosition().z;

                std::map<uint16_t, Ptr<UeManager>> UeMap = iter->second->GetRrc ()->m_ueMap;
                
                for (std::map<uint16_t, Ptr<UeManager>>::iterator iter2 = UeMap.begin(); iter2 != UeMap.end(); ++iter2){
                    uint64_t Imsi = iter2->second->GetImsi();
                    Imsi_List.push_back(Imsi);
                }

                double distance_sum = 0;
                double far_distance_ues = 0;
                for (uint64_t i=0; i<Imsi_List.size(); i++){
                    Ptr<LteUeNetDevice> UeNetDevice = m_ues[Imsi_List[i]];
                    double Ue_x = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetPosition().x;
                    double Ue_y = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetPosition().y;
                    double Ue_z = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetPosition().z;

                    double distance = sqrt( pow(eNB_x-Ue_x, 2) + pow(eNB_y-Ue_y, 2) + pow(eNB_z-Ue_z, 2) );
                    distance_sum += distance;

                    if (distance >= 237.74){
                        far_distance_ues += 1;
                    }
                }
                
                double distance_avg;
                float ratio_far_ues;
                float ratio_ues;
                float served_ues;

                if (Imsi_List.size() == 0){
                    distance_avg = 0;
                    ratio_far_ues = 0;
                    ratio_ues = 0;
                    served_ues = 0;
                }
                else {
                    distance_avg = double(distance_sum) / double(Imsi_List.size());
                    ratio_far_ues = float( far_distance_ues / Imsi_List.size() );
                    ratio_ues = float(Imsi_List.size()) / m_userCount;
                    served_ues = float(Imsi_List.size());
                }
                std::cout<<"eNB "<<CellId<<": "<<"Average Distance : "<<distance_avg<<std::endl;
                std::cout<<"eNB "<<CellId<<": "<<"Raito_of_far_ues : "<<ratio_far_ues<<std::endl;
                std::cout<<"eNB "<<CellId<<": "<<"Raito_of_served_ues : "<<ratio_ues<<std::endl;
                
                std::cout<<"---"<<std::endl;                
                ratio_ues = round(ratio_ues * 100) / 100;
                ratio_far_ues = round(ratio_far_ues * 100) / 100;
                
                Served_UEs.push_back(served_ues);
                Far_UEs.push_back(ratio_far_ues);
            }

            box = CreateObject < OpenGymBoxContainer < float > > (shape);
            box -> SetData(Served_UEs);
            obsContainer -> Add("ServedUes", box);

            box = CreateObject < OpenGymBoxContainer < float > > (shape);
            box -> SetData(Far_UEs);
            obsContainer -> Add("FarUes", box);

            // DL Throughput
            dlThroughput_IMSI = RlcStats->GetdlThroughput_IMSI();
      
            std::vector <float> throughput(m_cellCount);
            for (std::map<uint64_t, Ptr<LteUeNetDevice>>::iterator iter = m_ues.begin(); iter != m_ues.end(); ++iter)
            {
                uint16_t CellId = iter->second->GetRrc()->GetCellId();
                uint32_t dlThroughput = dlThroughput_IMSI[iter->first];
                throughput[CellId-1] = throughput[CellId-1] + ( double(dlThroughput) * 8.0 / 1000000.0);


            }
            float throughput_total = 0.0;
            throughput_total = std::accumulate(throughput.begin (), throughput.end (), 0.0);
            
            std::cout <<"Custom Total Throughput : "<< throughput_total << std::endl;

            int i = 1;
            for(auto loop : throughput){
                std::cout<<"Cell "<<i<<"  Custom Throughtput : "<<loop<<std::endl;
                i += 1;
            }
            std::cout<<"**************************"<<std::endl;

            box = CreateObject < OpenGymBoxContainer < float > > (shape);
            box -> SetData(throughput);
            obsContainer -> Add("Throughput", box);

            // Average CQI
            box = CreateObject < OpenGymBoxContainer < float > > (shape);

            std::vector <float> TotalCqi(m_cellCount);
            float CqiSum = 0;

            for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter){
                uint32_t CellId = iter->first;
                std::vector<uint64_t> Imsi_List;

                std::map<uint16_t, Ptr<UeManager>> UeMap = iter->second->GetRrc ()->m_ueMap;

                for (std::map<uint16_t, Ptr<UeManager>>::iterator iter2 = UeMap.begin(); iter2 != UeMap.end(); ++iter2){
                    uint64_t Imsi = iter2->second->GetImsi();
                    Imsi_List.push_back(Imsi);
                }

                float AvgCqi = 0;
                for (uint64_t i=0; i<Imsi_List.size(); i++)
                {
                    float UeCqi = float(m_ues[Imsi_List[i]]->GetPhy()->AvgCqi);
                    AvgCqi += UeCqi;
                }

                
                if(Imsi_List.size() == 0){
                    AvgCqi = 0;
                    
                    CqiSum = 0;
                }
                else{
                    std::cout<<"Cell Id: "<<CellId<<"  CQI SUM: "<<AvgCqi<<std::endl;
                    CqiSum += AvgCqi;
                    
                    AvgCqi = AvgCqi / Imsi_List.size(); 
                }

                AvgCqi = round(AvgCqi * 100) / 100;

                box -> AddValue(AvgCqi);
                std::cout<<"Cell ID: "<<CellId<<"  Average CQI: "<<AvgCqi<<std::endl;
            }
            obsContainer -> Add("AvgCqi", box);

            box = CreateObject < OpenGymBoxContainer < float > > (shape);
            CqiSum = CqiSum / m_ues.size();
            for (uint8_t idx = 0; idx < m_cellCount; ++idx) {
                TotalCqi[idx] = CqiSum;
            }
            box -> SetData(TotalCqi);
            obsContainer -> Add("TotalCqi",box);
            
            ///////////////////////////

            NS_LOG_LOGIC("MyGetObservation: " << obsContainer);
            std::cout<<"**************************"<<std::endl;
        
            return obsContainer;
        }

    void
    MyGymEnv::resetObs() {
        NS_LOG_FUNCTION(this);
        m_rbUtil.assign(m_cellCount, 0);
        rewards.assign(m_cellCount, 0);
        rewards_sum.assign(m_cellCount, 0);
        m_cellFrequency.assign(m_cellCount, 0);
        m_dlThroughputVec.assign(m_cellCount, 0);
        m_dlThroughput = 0;
        UserThrouput.clear();
        m_UesNum.assign(m_cellCount, 0);
        std::vector < uint32_t > dummyVec(29, 0);
        m_MCSPen.assign(m_cellCount, dummyVec);
        NS_LOG_LOGIC("%%%%%%%% Stop collecting %%%%%%%%  time= " << Simulator::Now().GetSeconds() << " sec");
        collect = 0;
        Simulator::Schedule(Seconds(m_interval - collecting_window), & MyGymEnv::Start_Collecting, this);
    }

    void
    MyGymEnv::calculate_rewards() {
        std::map < uint16_t, std::map < uint16_t, float >> ::iterator itr;

        std::map < uint16_t, float > ::iterator ptr;
        int Blocked_Users_num = 0;
        double min_thro = 100;
	    m_UesNum.assign(m_cellCount, 0);
        float sum_reward=0 ;

        for (itr = UserThrouput.begin(); itr != UserThrouput.end(); itr++) {

            float all = 0;
            std::map < uint16_t, float > tempmap = itr -> second;
            m_UesNum.at(itr -> first - 1) = tempmap.size();
            
            NS_LOG_LOGIC("Cell: " << itr -> first << " total throuput: " << m_dlThroughputVec.at(itr -> first - 1));
            NS_LOG_LOGIC("#users : " << tempmap.size());
            for (ptr = itr -> second.begin(); ptr != itr -> second.end(); ptr++) {
                if (ptr -> second < block_Thr)
                    Blocked_Users_num++;
                if (ptr -> second < min_thro)
                    min_thro = ptr -> second;
                NS_LOG_LOGIC("rnti: " << ptr -> first << " throughput:  " << ptr -> second);
                all = all + ptr -> second;
            }

            NS_LOG_LOGIC("using sum Cell: " << itr -> first << " total throuput: " << all);
            rewards.at(itr -> first - 1) = all;
            sum_reward += all;
        }

        for (itr = UserThrouput.begin(); itr != UserThrouput.end(); itr++) {
            rewards_sum.at(itr->first - 1) = sum_reward + rewards.at(itr->first -1);
        }
    }

    float
    MyGymEnv::GetReward() {
        NS_LOG_FUNCTION(this);

        float reward = 0;
    
        resetObs();
        return reward;
    }

    std::string
    MyGymEnv::GetExtraInfo() {
        NS_LOG_FUNCTION(this);
        return "";
    }

    bool
    MyGymEnv::ExecuteActions(Ptr < OpenGymDataContainer > action) {
        
        Ptr<OpenGymBoxContainer<float> > box = DynamicCast<OpenGymBoxContainer<float>>(action);

        uint32_t nodeNum = m_enbs.size();

        std::list<LteRrcSap::CellsToAddMod> celllist_temp;
        std::vector <double> Power_list;

        for(uint32_t i = 0; i < nodeNum; i++){
            LteRrcSap::CellsToAddMod cell_temp;
            cell_temp.cellIndex = i+1;
            cell_temp.physCellId = 0;
            cell_temp.cellIndividualOffset = (int8_t)box->GetValue(i);

            std::cout<<"CIO: "<<box->GetValue(i)<<std::endl;
            
            celllist_temp.push_back(cell_temp);
        }
        for(uint32_t i = 0; i < nodeNum; i++){
            
            // Power Control
            Power_list.push_back(32+((double)box->GetValue(i+5)));
        
            // Non-Power Control
            // Power_list.push_back(32);
        }
        for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter){
            iter->second->m_rrc->setCellstoAddModList(celllist_temp);

            uint32_t idx = iter->second->GetCellId();
            idx = idx-1;
            std::cout << "CellId :   " << iter->second->GetCellId() << std::endl;
            ns3::Ptr<ns3::LteEnbPhy> enbPhy = iter->second->GetPhy();
            enbPhy->SetTxPower(Power_list.at(idx)); // Apply TXP action
            NS_LOG_UNCOND (" tx power: " << enbPhy->GetTxPower());

            std::map<uint16_t, Ptr<UeManager>> m_UeMap = iter->second->GetRrc()->m_ueMap;
            for(auto iter2 = m_UeMap.begin(); iter2 != m_UeMap.end(); iter2++)
            {
                iter2->second->ScheduleRrcConnectionRecursive(); // Apply CIO action
            }
        }
        std::cout<<"**********************************************"<<std::endl;

        return true;
    }

    uint8_t
    MyGymEnv::Convert2ITB(uint8_t mcsIdx) {
        uint8_t iTBS;
        if (mcsIdx < 10) {
            iTBS = mcsIdx;
        } else if (mcsIdx < 17) {
            iTBS = mcsIdx - 1;
        } else {
            iTBS = mcsIdx - 2;
        }

        return iTBS;
    }

    uint8_t
    MyGymEnv::GetnRB(uint8_t iTB, uint16_t tbSize) {
        uint32_t tbSizeb = uint32_t(tbSize) * uint32_t(8);
        // search in list
    uint32_t tbList[]  = {
			  16,32,56,88,120,152,176,208,224,256,288,328,344,376,392,424,456,488,504,536,568,600,616,648,680,712,744,776,776,808,840,872,904,936,968,1000,1032,1032,1064,1096,1128,1160,1192,1224,1256,1256,1288,1320,1352,1384,1416,1416,1480,1480,1544,1544,1608,1608,1608,1672,1672,1736,1736,1800,1800,1800,1864,1864,1928,1928,1992,1992,2024,2088,2088,2088,2152,2152,2216,2216,2280,2280,2280,2344,2344,2408,2408,2472,2472,2536,2536,2536,2600,2600,2664,2664,2728,2728,2728,2792,2792,2856,2856,2856,2984,2984,2984,2984,2984,3112,
		24,56,88,144,176,208,224,256,328,344,376,424,456,488,520,568,600,632,680,712,744,776,808,872,904,936,968,1000,1032,1064,1128,1160,1192,1224,1256,1288,1352,1384,1416,1416,1480,1544,1544,1608,1608,1672,1736,1736,1800,1800,1864,1864,1928,1992,1992,2024,2088,2088,2152,2152,2216,2280,2280,2344,2344,2408,2472,2472,2536,2536,2600,2600,2664,2728,2728,2792,2792,2856,2856,2856,2984,2984,2984,3112,3112,3112,3240,3240,3240,3240,3368,3368,3368,3496,3496,3496,3496,3624,3624,3624,3752,3752,3752,3752,3880,3880,3880,4008,4008,4008,
		32,72,144,176,208,256,296,328,376,424,472,520,568,616,648,696,744,776,840,872,936,968,1000,1064,1096,1160,1192,1256,1288,1320,1384,1416,1480,1544,1544,1608,1672,1672,1736,1800,1800,1864,1928,1992,2024,2088,2088,2152,2216,2216,2280,2344,2344,2408,2472,2536,2536,2600,2664,2664,2728,2792,2856,2856,2856,2984,2984,3112,3112,3112,3240,3240,3240,3368,3368,3368,3496,3496,3496,3624,3624,3624,3752,3752,3880,3880,3880,4008,4008,4008,4136,4136,4136,4264,4264,4264,4392,4392,4392,4584,4584,4584,4584,4584,4776,4776,4776,4776,4968,4968,
		40,104,176,208,256,328,392,440,504,568,616,680,744,808,872,904,968,1032,1096,1160,1224,1256,1320,1384,1416,1480,1544,1608,1672,1736,1800,1864,1928,1992,2024,2088,2152,2216,2280,2344,2408,2472,2536,2536,2600,2664,2728,2792,2856,2856,2984,2984,3112,3112,3240,3240,3368,3368,3496,3496,3624,3624,3624,3752,3752,3880,3880,4008,4008,4136,4136,4264,4264,4392,4392,4392,4584,4584,4584,4776,4776,4776,4776,4968,4968,4968,5160,5160,5160,5352,5352,5352,5352,5544,5544,5544,5736,5736,5736,5736,5992,5992,5992,5992,6200,6200,6200,6200,6456,6456,
		56,120,208,256,328,408,488,552,632,696,776,840,904,1000,1064,1128,1192,1288,1352,1416,1480,1544,1608,1736,1800,1864,1928,1992,2088,2152,2216,2280,2344,2408,2472,2600,2664,2728,2792,2856,2984,2984,3112,3112,3240,3240,3368,3496,3496,3624,3624,3752,3752,3880,4008,4008,4136,4136,4264,4264,4392,4392,4584,4584,4584,4776,4776,4968,4968,4968,5160,5160,5160,5352,5352,5544,5544,5544,5736,5736,5736,5992,5992,5992,5992,6200,6200,6200,6456,6456,6456,6456,6712,6712,6712,6968,6968,6968,6968,7224,7224,7224,7480,7480,7480,7480,7736,7736,7736,7992,
		72,144,224,328,424,504,600,680,776,872,968,1032,1128,1224,1320,1384,1480,1544,1672,1736,1864,1928,2024,2088,2216,2280,2344,2472,2536,2664,2728,2792,2856,2984,3112,3112,3240,3368,3496,3496,3624,3752,3752,3880,4008,4008,4136,4264,4392,4392,4584,4584,4776,4776,4776,4968,4968,5160,5160,5352,5352,5544,5544,5736,5736,5736,5992,5992,5992,6200,6200,6200,6456,6456,6712,6712,6712,6968,6968,6968,7224,7224,7224,7480,7480,7480,7736,7736,7736,7992,7992,7992,8248,8248,8248,8504,8504,8760,8760,8760,8760,9144,9144,9144,9144,9528,9528,9528,9528,9528,
		328,176,256,392,504,600,712,808,936,1032,1128,1224,1352,1480,1544,1672,1736,1864,1992,2088,2216,2280,2408,2472,2600,2728,2792,2984,2984,3112,3240,3368,3496,3496,3624,3752,3880,4008,4136,4136,4264,4392,4584,4584,4776,4776,4968,4968,5160,5160,5352,5352,5544,5736,5736,5992,5992,5992,6200,6200,6456,6456,6456,6712,6712,6968,6968,6968,7224,7224,7480,7480,7736,7736,7736,7992,7992,8248,8248,8248,8504,8504,8760,8760,8760,9144,9144,9144,9144,9528,9528,9528,9528,9912,9912,9912,10296,10296,10296,10296,10680,10680,10680,10680,11064,11064,11064,11448,11448,11448,
		104,224,328,472,584,712,840,968,1096,1224,1320,1480,1608,1672,1800,1928,2088,2216,2344,2472,2536,2664,2792,2984,3112,3240,3368,3368,3496,3624,3752,3880,4008,4136,4264,4392,4584,4584,4776,4968,4968,5160,5352,5352,5544,5736,5736,5992,5992,6200,6200,6456,6456,6712,6712,6712,6968,6968,7224,7224,7480,7480,7736,7736,7992,7992,8248,8248,8504,8504,8760,8760,8760,9144,9144,9144,9528,9528,9528,9912,9912,9912,10296,10296,10296,10680,10680,10680,11064,11064,11064,11448,11448,11448,11448,11832,11832,11832,12216,12216,12216,12576,12576,12576,12960,12960,12960,12960,13536,13536,
		120,256,392,536,680,808,968,1096,1256,1384,1544,1672,1800,1928,2088,2216,2344,2536,2664,2792,2984,3112,3240,3368,3496,3624,3752,3880,4008,4264,4392,4584,4584,4776,4968,4968,5160,5352,5544,5544,5736,5992,5992,6200,6200,6456,6456,6712,6968,6968,7224,7224,7480,7480,7736,7736,7992,7992,8248,8504,8504,8760,8760,9144,9144,9144,9528,9528,9528,9912,9912,9912,10296,10296,10680,10680,10680,11064,11064,11064,11448,11448,11448,11832,11832,12216,12216,12216,12576,12576,12576,12960,12960,12960,13536,13536,13536,13536,14112,14112,14112,14112,14688,14688,14688,14688,15264,15264,15264,15264,
		136,296,456,616,776,936,1096,1256,1416,1544,1736,1864,2024,2216,2344,2536,2664,2856,2984,3112,3368,3496,3624,3752,4008,4136,4264,4392,4584,4776,4968,5160,5160,5352,5544,5736,5736,5992,6200,6200,6456,6712,6712,6968,6968,7224,7480,7480,7736,7992,7992,8248,8248,8504,8760,8760,9144,9144,9144,9528,9528,9912,9912,10296,10296,10296,10680,10680,11064,11064,11064,11448,11448,11832,11832,11832,12216,12216,12576,12576,12960,12960,12960,13536,13536,13536,13536,14112,14112,14112,14112,14688,14688,14688,15264,15264,15264,15264,15840,15840,15840,16416,16416,16416,16416,16992,16992,16992,16992,17568,
		144,328,504,680,872,1032,1224,1384,1544,1736,1928,2088,2280,2472,2664,2792,2984,3112,3368,3496,3752,3880,4008,4264,4392,4584,4776,4968,5160,5352,5544,5736,5736,5992,6200,6200,6456,6712,6712,6968,7224,7480,7480,7736,7992,7992,8248,8504,8504,8760,9144,9144,9144,9528,9528,9912,9912,10296,10296,10680,10680,11064,11064,11448,11448,11448,11832,11832,12216,12216,12576,12576,12960,12960,12960,13536,13536,13536,14112,14112,14112,14688,14688,14688,14688,15264,15264,15264,15840,15840,15840,16416,16416,16416,16992,16992,16992,16992,17568,17568,17568,18336,18336,18336,18336,18336,19080,19080,19080,19080,
		176,376,584,776,1000,1192,1384,1608,1800,2024,2216,2408,2600,2792,2984,3240,3496,3624,3880,4008,4264,4392,4584,4776,4968,5352,5544,5736,5992,5992,6200,6456,6712,6968,6968,7224,7480,7736,7736,7992,8248,8504,8760,8760,9144,9144,9528,9528,9912,9912,10296,10680,10680,11064,11064,11448,11448,11832,11832,12216,12216,12576,12576,12960,12960,13536,13536,13536,14112,14112,14112,14688,14688,14688,15264,15264,15840,15840,15840,16416,16416,16416,16992,16992,16992,17568,17568,17568,18336,18336,18336,18336,19080,19080,19080,19080,19848,19848,19848,19848,20616,20616,20616,21384,21384,21384,21384,22152,22152,22152,
		208,440,680,904,1128,1352,1608,1800,2024,2280,2472,2728,2984,3240,3368,3624,3880,4136,4392,4584,4776,4968,5352,5544,5736,5992,6200,6456,6712,6712,6968,7224,7480,7736,7992,8248,8504,8760,8760,9144,9528,9528,9912,9912,10296,10680,10680,11064,11064,11448,11832,11832,12216,12216,12576,12576,12960,12960,13536,13536,14112,14112,14112,14688,14688,15264,15264,15264,15840,15840,16416,16416,16416,16992,16992,17568,17568,17568,18336,18336,18336,19080,19080,19080,19080,19848,19848,19848,20616,20616,20616,21384,21384,21384,21384,22152,22152,22152,22920,22920,22920,23688,23688,23688,23688,24496,24496,24496,24496,25456,
		224,488,744,1000,1256,1544,1800,2024,2280,2536,2856,3112,3368,3624,3880,4136,4392,4584,4968,5160,5352,5736,5992,6200,6456,6712,6968,7224,7480,7736,7992,8248,8504,8760,9144,9144,9528,9912,9912,10296,10680,10680,11064,11448,11448,11832,12216,12216,12576,12960,12960,13536,13536,14112,14112,14688,14688,14688,15264,15264,15840,15840,16416,16416,16992,16992,16992,17568,17568,18336,18336,18336,19080,19080,19080,19848,19848,19848,20616,20616,20616,21384,21384,21384,22152,22152,22152,22920,22920,22920,23688,23688,23688,24496,24496,24496,25456,25456,25456,25456,26416,26416,26416,26416,27376,27376,27376,27376,28336,28336,
		256,552,840,1128,1416,1736,1992,2280,2600,2856,3112,3496,3752,4008,4264,4584,4968,5160,5544,5736,5992,6200,6456,6968,7224,7480,7736,7992,8248,8504,8760,9144,9528,9912,9912,10296,10680,11064,11064,11448,11832,12216,12216,12576,12960,12960,13536,13536,14112,14112,14688,14688,15264,15264,15840,15840,16416,16416,16992,16992,17568,17568,18336,18336,18336,19080,19080,19848,19848,19848,20616,20616,20616,21384,21384,22152,22152,22152,22920,22920,22920,23688,23688,24496,24496,24496,25456,25456,25456,25456,26416,26416,26416,27376,27376,27376,28336,28336,28336,28336,29296,29296,29296,29296,30576,30576,30576,30576,31704,31704,
		280,600,904,1224,1544,1800,2152,2472,2728,3112,3368,3624,4008,4264,4584,4968,5160,5544,5736,6200,6456,6712,6968,7224,7736,7992,8248,8504,8760,9144,9528,9912,10296,10296,10680,11064,11448,11832,11832,12216,12576,12960,12960,13536,13536,14112,14688,14688,15264,15264,15840,15840,16416,16416,16992,16992,17568,17568,18336,18336,18336,19080,19080,19848,19848,20616,20616,20616,21384,21384,22152,22152,22152,22920,22920,23688,23688,23688,24496,24496,24496,25456,25456,25456,26416,26416,26416,27376,27376,27376,28336,28336,28336,29296,29296,29296,29296,30576,30576,30576,30576,31704,31704,31704,31704,32856,32856,32856,34008,34008,
		328,632,968,1288,1608,1928,2280,2600,2984,3240,3624,3880,4264,4584,4968,5160,5544,5992,6200,6456,6712,7224,7480,7736,7992,8504,8760,9144,9528,9912,9912,10296,10680,11064,11448,11832,12216,12216,12576,12960,13536,13536,14112,14112,14688,14688,15264,15840,15840,16416,16416,16992,16992,17568,17568,18336,18336,19080,19080,19848,19848,19848,20616,20616,21384,21384,22152,22152,22152,22920,22920,23688,23688,24496,24496,24496,25456,25456,25456,26416,26416,26416,27376,27376,27376,28336,28336,28336,29296,29296,29296,30576,30576,30576,30576,31704,31704,31704,31704,32856,32856,32856,34008,34008,34008,34008,35160,35160,35160,35160,
		336,696,1064,1416,1800,2152,2536,2856,3240,3624,4008,4392,4776,5160,5352,5736,6200,6456,6712,7224,7480,7992,8248,8760,9144,9528,9912,10296,10296,10680,11064,11448,11832,12216,12576,12960,13536,13536,14112,14688,14688,15264,15264,15840,16416,16416,16992,17568,17568,18336,18336,19080,19080,19848,19848,20616,20616,20616,21384,21384,22152,22152,22920,22920,23688,23688,24496,24496,24496,25456,25456,26416,26416,26416,27376,27376,27376,28336,28336,29296,29296,29296,30576,30576,30576,30576,31704,31704,31704,32856,32856,32856,34008,34008,34008,35160,35160,35160,35160,36696,36696,36696,36696,37888,37888,37888,39232,39232,39232,39232,
		376,776,1160,1544,1992,2344,2792,3112,3624,4008,4392,4776,5160,5544,5992,6200,6712,7224,7480,7992,8248,8760,9144,9528,9912,10296,10680,11064,11448,11832,12216,12576,12960,13536,14112,14112,14688,15264,15264,15840,16416,16416,16992,17568,17568,18336,18336,19080,19080,19848,19848,20616,21384,21384,22152,22152,22920,22920,23688,23688,24496,24496,24496,25456,25456,26416,26416,27376,27376,27376,28336,28336,29296,29296,29296,30576,30576,30576,31704,31704,31704,32856,32856,32856,34008,34008,34008,35160,35160,35160,36696,36696,36696,37888,37888,37888,37888,39232,39232,39232,40576,40576,40576,40576,42368,42368,42368,42368,43816,43816,
		408,840,1288,1736,2152,2600,2984,3496,3880,4264,4776,5160,5544,5992,6456,6968,7224,7736,8248,8504,9144,9528,9912,10296,10680,11064,11448,12216,12576,12960,13536,13536,14112,14688,15264,15264,15840,16416,16992,16992,17568,18336,18336,19080,19080,19848,20616,20616,21384,21384,22152,22152,22920,22920,23688,24496,24496,25456,25456,25456,26416,26416,27376,27376,28336,28336,29296,29296,29296,30576,30576,30576,31704,31704,32856,32856,32856,34008,34008,34008,35160,35160,35160,36696,36696,36696,37888,37888,37888,39232,39232,39232,40576,40576,40576,40576,42368,42368,42368,43816,43816,43816,43816,45352,45352,45352,46888,46888,46888,46888,
		440,904,1384,1864,2344,2792,3240,3752,4136,4584,5160,5544,5992,6456,6968,7480,7992,8248,8760,9144,9912,10296,10680,11064,11448,12216,12576,12960,13536,14112,14688,14688,15264,15840,16416,16992,16992,17568,18336,18336,19080,19848,19848,20616,20616,21384,22152,22152,22920,22920,23688,24496,24496,25456,25456,26416,26416,27376,27376,28336,28336,29296,29296,29296,30576,30576,31704,31704,31704,32856,32856,34008,34008,34008,35160,35160,35160,36696,36696,36696,37888,37888,39232,39232,39232,40576,40576,40576,42368,42368,42368,42368,43816,43816,43816,45352,45352,45352,46888,46888,46888,46888,48936,48936,48936,48936,48936,51024,51024,51024,
		488,1000,1480,1992,2472,2984,3496,4008,4584,4968,5544,5992,6456,6968,7480,7992,8504,9144,9528,9912,10680,11064,11448,12216,12576,12960,13536,14112,14688,15264,15840,15840,16416,16992,17568,18336,18336,19080,19848,19848,20616,21384,21384,22152,22920,22920,23688,24496,24496,25456,25456,26416,26416,27376,27376,28336,28336,29296,29296,30576,30576,31704,31704,31704,32856,32856,34008,34008,35160,35160,35160,36696,36696,36696,37888,37888,39232,39232,39232,40576,40576,40576,42368,42368,42368,43816,43816,43816,45352,45352,45352,46888,46888,46888,46888,48936,48936,48936,48936,51024,51024,51024,51024,52752,52752,52752,52752,55056,55056,55056,
		520,1064,1608,2152,2664,3240,3752,4264,4776,5352,5992,6456,6968,7480,7992,8504,9144,9528,10296,10680,11448,11832,12576,12960,13536,14112,14688,15264,15840,16416,16992,16992,17568,18336,19080,19080,19848,20616,21384,21384,22152,22920,22920,23688,24496,24496,25456,25456,26416,27376,27376,28336,28336,29296,29296,30576,30576,31704,31704,32856,32856,34008,34008,34008,35160,35160,36696,36696,36696,37888,37888,39232,39232,40576,40576,40576,42368,42368,42368,43816,43816,43816,45352,45352,45352,46888,46888,46888,48936,48936,48936,48936,51024,51024,51024,51024,52752,52752,52752,55056,55056,55056,55056,57336,57336,57336,57336,59256,59256,59256,
		552,1128,1736,2280,2856,3496,4008,4584,5160,5736,6200,6968,7480,7992,8504,9144,9912,10296,11064,11448,12216,12576,12960,13536,14112,14688,15264,15840,16416,16992,17568,18336,19080,19848,19848,20616,21384,22152,22152,22920,23688,24496,24496,25456,25456,26416,27376,27376,28336,28336,29296,29296,30576,30576,31704,31704,32856,32856,34008,34008,35160,35160,36696,36696,37888,37888,37888,39232,39232,40576,40576,40576,42368,42368,43816,43816,43816,45352,45352,45352,46888,46888,46888,48936,48936,48936,51024,51024,51024,51024,52752,52752,52752,55056,55056,55056,55056,57336,57336,57336,57336,59256,59256,59256,59256,61664,61664,61664,61664,63776,
		584,1192,1800,2408,2984,3624,4264,4968,5544,5992,6712,7224,7992,8504,9144,9912,10296,11064,11448,12216,12960,13536,14112,14688,15264,15840,16416,16992,17568,18336,19080,19848,19848,20616,21384,22152,22920,22920,23688,24496,25456,25456,26416,26416,27376,28336,28336,29296,29296,30576,31704,31704,32856,32856,34008,34008,35160,35160,36696,36696,36696,37888,37888,39232,39232,40576,40576,42368,42368,42368,43816,43816,45352,45352,45352,46888,46888,46888,48936,48936,48936,51024,51024,51024,52752,52752,52752,52752,55056,55056,55056,57336,57336,57336,57336,59256,59256,59256,61664,61664,61664,61664,63776,63776,63776,63776,66592,66592,66592,66592,
		616,1256,1864,2536,3112,3752,4392,5160,5736,6200,6968,7480,8248,8760,9528,10296,10680,11448,12216,12576,13536,14112,14688,15264,15840,16416,16992,17568,18336,19080,19848,20616,20616,21384,22152,22920,23688,24496,24496,25456,26416,26416,27376,28336,28336,29296,29296,30576,31704,31704,32856,32856,34008,34008,35160,35160,36696,36696,37888,37888,39232,39232,40576,40576,40576,42368,42368,43816,43816,43816,45352,45352,46888,46888,46888,48936,48936,48936,51024,51024,51024,52752,52752,52752,55056,55056,55056,55056,57336,57336,57336,59256,59256,59256,61664,61664,61664,61664,63776,63776,63776,63776,66592,66592,66592,66592,68808,68808,68808,71112,
		712,1480,2216,2984,3752,4392,5160,5992,6712,7480,8248,8760,9528,10296,11064,11832,12576,13536,14112,14688,15264,16416,16992,17568,18336,19080,19848,20616,21384,22152,22920,23688,24496,25456,25456,26416,27376,28336,29296,29296,30576,30576,31704,32856,32856,34008,35160,35160,36696,36696,37888,37888,39232,40576,40576,40576,42368,42368,43816,43816,45352,45352,46888,46888,48936,48936,48936,51024,51024,52752,52752,52752,55056,55056,55056,55056,57336,57336,57336,59256,59256,59256,61664,61664,61664,63776,63776,63776,66592,66592,66592,68808,68808,68808,71112,71112,71112,73712,73712,75376,75376,75376,75376,75376,75376,75376,75376,75376,75376,75376};

        uint16_t count = iTB * 110;
        while (tbList[count++] != tbSizeb);

        return (count % 110);

    }

    void
    MyGymEnv::GetPhyStats(Ptr < MyGymEnv > gymEnv,
        const PhyTransmissionStatParameters params) {
        if (gymEnv -> collect == 1) {
            NS_LOG_LOGIC(" ======= New Transmission =======");
            NS_LOG_LOGIC("Packed sent by" << " cell: " << params.m_cellId << " to UE: " << params.m_rnti);
            // Get size of TB
            uint32_t idx = params.m_cellId - 1;
            gymEnv -> m_cellFrequency.at(idx) = gymEnv -> m_cellFrequency.at(idx) + 1;
            gymEnv -> m_dlThroughputVec.at(idx) = gymEnv -> m_dlThroughputVec.at(idx) + (params.m_size) * 8.0 / 1024.0 / 1024.0 / gymEnv -> collecting_window;
            gymEnv -> m_dlThroughput = gymEnv -> m_dlThroughput + (params.m_size) * 8.0 / 1024.0 / 1024.0 / gymEnv -> collecting_window;
            NS_LOG_LOGIC("sent data at cell " << idx << " is " << gymEnv -> m_dlThroughputVec.at(idx) * gymEnv -> collecting_window);

            //add throughput per user
            gymEnv -> UserThrouput[params.m_cellId][params.m_rnti] = gymEnv -> UserThrouput[params.m_cellId][params.m_rnti] + (params.m_size) * 8.0 / 1024.0 / 1024.0 / gymEnv -> collecting_window;

            // Get nRBs
            uint8_t nRBs = MyGymEnv::GetnRB(MyGymEnv::Convert2ITB(params.m_mcs), params.m_size);
            gymEnv -> m_rbUtil.at(idx) = gymEnv -> m_rbUtil.at(idx) + nRBs;

            // Get MCSPen
            gymEnv -> m_MCSPen.at(idx).at(params.m_mcs) = gymEnv -> m_MCSPen.at(idx).at(params.m_mcs) + 1;
            NS_LOG_LOGIC("Frequency at cell " << idx << " is " << gymEnv -> m_cellFrequency.at(idx));
            NS_LOG_LOGIC("DLThroughput at cell " << idx << " is " << gymEnv -> m_dlThroughputVec.at(idx));
            NS_LOG_LOGIC("NRB at cell " << idx << " is " << gymEnv -> m_rbUtil.at(idx));
            NS_LOG_LOGIC("MCS frequency at cell " << idx << " of MCS " << uint32_t(params.m_mcs) << " is " << gymEnv -> m_MCSPen[idx][params.m_mcs]);
        }
    }

} // ns3 namespace

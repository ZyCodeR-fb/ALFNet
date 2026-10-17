<table>
  <thead>
    <tr>
      <th rowspan="2" scope="col">模型</th>
      <th colspan="3" scope="colgroup">METR-LA</th>
      <th colspan="3" scope="colgroup">PEMS-BAY</th>
      <th colspan="3" scope="colgroup">PEMS04</th>
      <th colspan="3" scope="colgroup">PEMS08</th>
    </tr>
    <tr>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ALFNet</td>
      <td><b>3.842</b></td><td><b>6.085</b></td><td><b>10.337</b></td>
      <td><b>2.723</b></td><td><b>4.201</b></td><td><b>6.276</b></td>
      <td><b>39.565</b></td><td><u>53.346</u></td><td><b>23.257</b></td>
      <td><u>36.780</u></td><td><u>46.567</u></td><td><b>43.000</b></td>
    </tr>
    <tr>
      <td>w/o State Age</td>
      <td>3.878</td><td><u>6.097</u></td><td>10.423</td>
      <td>2.732</td><td><u>4.248</u></td><td><u>6.464</u></td>
      <td><u>39.757</u></td><td><b>53.136</b></td><td>24.957</td>
      <td>38.162</td><td>47.792</td><td>48.701</td>
    </tr>
    <tr>
      <td>w/o Adaptive Routing</td>
      <td>3.920</td><td>6.127</td><td>10.602</td>
      <td>2.770</td><td>4.280</td><td>6.514</td>
      <td>39.826</td><td>53.401</td><td><u>23.415</u></td>
      <td>38.783</td><td>49.591</td><td>58.378</td>
    </tr>
    <tr>
      <td>w/o Spatial Context</td>
      <td>4.400</td><td>6.645</td><td>12.529</td>
      <td>2.958</td><td>4.493</td><td>7.011</td>
      <td>40.292</td><td>54.694</td><td>24.877</td>
      <td>42.382</td><td>53.475</td><td>53.061</td>
    </tr>
    <tr>
      <td>w/o ALFTemporal–FiLM</td>
      <td>4.264</td><td>6.560</td><td>11.760</td>
      <td>2.890</td><td>4.487</td><td>6.992</td>
      <td>42.137</td><td>56.293</td><td>26.361</td>
      <td>40.906</td><td>51.874</td><td>52.003</td>
    </tr>
    <tr>
      <td>w/o Graph Refinement</td>
      <td><u>3.852</u></td><td>6.106</td><td><u>10.385</u></td>
      <td><u>2.726</u></td><td>4.258</td><td>6.525</td>
      <td>40.032</td><td>53.743</td><td>23.509</td>
      <td><b>35.275</b></td><td><b>45.048</b></td><td><u>43.313</u></td>
    </tr>
  </tbody>
</table>

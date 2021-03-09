// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import Head from "next/head";
import React from "react";
import _ from "lodash";
import { Grid, Box } from "@material-ui/core";
import { Header } from "../../../components/authenticated/header/Header";
import { Sidebar } from "../../../components/authenticated/sidebar/Sidebar";
import { CustomerTable } from "../../../components/admin/customer-table/CustomerTable";

const Admin = () => {
  return (
    <main>
      <Head>
        <title>Admin - Amazon SFS</title>
      </Head>
      <Grid container>
        <Grid item xs={12} md={3} lg={2}>
          <Sidebar />
        </Grid>
        <Grid item xs={12} md={9} lg={10}>
          <Box height="100%" display="flex" flexDirection="column">
            <Header>Customers</Header>
            <CustomerTable />
          </Box>
        </Grid>
      </Grid>
    </main>
  );
};

export default Admin;

// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import Head from "next/head";
import { Grid } from "@material-ui/core";
import { Sidebar } from "../../../components/authenticated/sidebar/Sidebar";
import { Upload as UploadForm } from "../../../components/authenticated/upload/Upload";

const Upload = () => {
  return (
    <main>
      <Head>
        <title>New Report - Amazon SFS</title>
      </Head>
      <Grid container>
        <Grid item xs={12} md={3}>
          <Sidebar />
        </Grid>
        <Grid item xs={12} md={9}>
          <UploadForm />
        </Grid>
      </Grid>
    </main>
  );
};

export default Upload;

// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";

const theme = createMuiTheme({
  typography: {
    fontFamily: ["Rubik", "sans-serif"].join(","),
  },
  palette: {
    primary: {
      main: "#ffaf20",
      contrastText: "#2f384d",
    },
    secondary: {
      main: "#BFC5D2",
      contrastText: "#2E384D",
    },
  },
});

export const VoyagerTheme = ({ children }) => {
  return <ThemeProvider theme={theme}>{children}</ThemeProvider>;
};
